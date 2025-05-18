import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import csv
from datetime import datetime
from math import sqrt
from skimage.transform import resize
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.models import inception_v3
from torchvision import transforms
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== MODEL COMPONENTS ======================
class PixelNormalization(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)

class MinibatchStdev(nn.Module):
    def forward(self, x):
        batch_size, _, h, w = x.size()
        stdev = torch.std(x, dim=0, keepdim=True)
        mean_stdev = torch.mean(stdev, dim=[1,2,3], keepdim=True)
        mean_stdev = mean_stdev.expand(batch_size, 1, h, w)
        return torch.cat([x, mean_stdev], dim=1)

class WeightedSum(nn.Module):
    def __init__(self, alpha=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
    def forward(self, inputs):
        return (1 - self.alpha) * inputs[0] + self.alpha * inputs[1]

class Conv2dWS(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        nn.init.normal_(self.weight, mean=0, std=1)
        self.scale = (2 / (in_channels * torch.prod(torch.tensor(kernel_size)))) ** 0.5
        if self.bias is not None:
            self.bias.data.zero_()
    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class ExponentialMovingAverage:
    def __init__(self, parameters, decay=0.999):
        self.parameters = list(parameters)
        self.decay = decay
        self.shadow_params = [p.clone().detach() for p in self.parameters]
    
    def update(self):
        for s_param, param in zip(self.shadow_params, self.parameters):
            s_param.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_to(self, model):
        for s_param, param in zip(self.shadow_params, model.parameters()):
            param.data.copy_(s_param)

# ====================== GENERATOR ======================
class ProgressiveGenerator(nn.Module):
    def __init__(self, latent_dim=512, depth=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.depth = depth

        self.initial = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.Unflatten(1, (128, 4, 4)),
            PixelNormalization(),
            Conv2dWS(128, 128, 3, padding=1),
            PixelNormalization(),
            nn.LeakyReLU(0.2),
            Conv2dWS(128, 128, 3, padding=1),
            PixelNormalization(),
            nn.LeakyReLU(0.2)
        )

        self.blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()

        for _ in range(depth):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                Conv2dWS(128, 128, 3, padding=1),
                PixelNormalization(),
                nn.LeakyReLU(0.2),
                Conv2dWS(128, 128, 3, padding=1),
                PixelNormalization(),
                nn.LeakyReLU(0.2)
            )
            self.blocks.append(block)
            self.to_rgb.append(Conv2dWS(128, 3, 1))

    def forward(self, x, step, alpha):
        x = self.initial(x)
        if step == 0:
            return self.to_rgb[0](x)
        for i in range(step-1):
            x = self.blocks[i](x)
        x_prev = x
        x = self.blocks[step-1](x)
        out_old = self.to_rgb[step-1](x_prev)
        out_old = F.interpolate(out_old, scale_factor=2, mode='nearest')
        out_new = self.to_rgb[step](x)
        return (1 - alpha) * out_old + alpha * out_new

# ====================== DISCRIMINATOR ======================
class ProgressiveDiscriminator(nn.Module):
    def __init__(self, n_blocks=6):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.from_rgb = nn.ModuleList()
        self.final_conv = nn.Sequential(
            MinibatchStdev(),
            Conv2dWS(128 + 1, 1, 1)
        )
        self._add_block(128, 128)
        for _ in range(1, n_blocks):
            self._add_block(128, 128)

    def _add_block(self, in_ch, out_ch):
        block = nn.Sequential(
            Conv2dWS(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2),
            Conv2dWS(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2)
        )
        self.blocks.insert(0, block)
        self.from_rgb.insert(0, Conv2dWS(3, in_ch, 1))

    def forward(self, x, step, alpha):
        x = self.from_rgb[-step-1](x)
        for i in range(step, len(self.blocks)):
            if x.shape[-2] <= 1 or x.shape[-1] <= 1:
                break
            x = self.blocks[i](x)
        x = self.final_conv(x)
        return x.view(-1)

# ====================== UTILITIES ======================
def scale_images(images, new_size):
    images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
    scaled = np.zeros((images_np.shape[0], new_size, new_size, 3))
    for i in range(images_np.shape[0]):
        img = resize(images_np[i], (new_size, new_size), mode='reflect', anti_aliasing=True)
        scaled[i] = img
    scaled = torch.from_numpy(scaled.transpose(0, 3, 1, 2)).float().to(device)
    return torch.clamp(scaled, -1, 1)

def load_real_samples(filename, max_samples):
    data = np.load(filename)
    X = data['faces'][:max_samples].astype('float32')
    print(f"Dataset range before normalization: {X.min()} to {X.max()}")
    return (X.transpose(0, 3, 1, 2) * 2) - 1

def setup_directories():
    os.makedirs('training_images', exist_ok=True)
    os.makedirs('training_metrics', exist_ok=True)
    os.makedirs('model_checkpoints', exist_ok=True)

def init_metrics_log():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = open(f'training_metrics/metrics_{timestamp}.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['Epoch', 'Stage', 'Resolution', 'D_Loss', 'G_Loss', 'Alpha'])
    return csv_file, writer

def compute_r1_penalty(d_out, x_real):
    grad_real = torch.autograd.grad(
        outputs=d_out.sum(),
        inputs=x_real,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    return grad_real.pow(2).reshape(grad_real.size(0), -1).sum(1).mean()


# ====================== FID METRIC ======================
inception_model = inception_v3(pretrained=True, transform_input=False).eval().to(device)

def compute_fid(real_images, generated_images):
    transform = transforms.Resize((299, 299))
    def get_activations(images):
        images = transform(images)
        images = adaptive_avg_pool2d(images, output_size=(299, 299))
        with torch.no_grad():
            pred = inception_model(images)
        return pred.detach().cpu().numpy()
    act1 = get_activations(real_images)
    act2 = get_activations(generated_images)
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# ====================== TRAIN FUNCTION ======================
def train(generator, discriminator, dataset, latent_dim=512, start_step=4):
    setup_directories()
    csv_file, metrics_writer = init_metrics_log()
    fid_log_file = open('training_metrics/fid_scores.csv', 'w', newline='')
    fid_writer = csv.writer(fid_log_file)
    fid_writer.writerow(['Step', 'Epoch', 'FID'])

    checkpoint_path = "model_checkpoints5/model_step2_stable.pth"
    checkpoint = torch.load(checkpoint_path)
    print(f"Checkpoint loaded keys: {list(checkpoint.keys())}")
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    g_optim = optim.Adam(generator.parameters(), lr=0.0005, betas=(0.0, 0.99))
    d_optim = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.0, 0.99))
    g_optim.load_state_dict(checkpoint['g_optim'])
    d_optim.load_state_dict(checkpoint['d_optim'])
    ema = ExponentialMovingAverage(generator.parameters())
    ema.shadow_params = checkpoint['ema']
    r1_weight = 0.01

    n_batch = [16, 16, 16, 8, 4, 4]
    n_epochs = [5, 8, 8, 10, 10, 10]
    all_d_losses = []
    all_g_losses = []
    fid_scores = []

    for step in range(start_step, len(n_epochs)):
        epochs_norm, epochs_fade, batch_size = n_epochs[step], n_epochs[step], n_batch[step]
        current_size = 4 * (2 ** step)

        for phase, phase_name, alpha in [(epochs_fade, 'Fade', lambda e, total: e / max(total - 1, 1)), (epochs_norm, 'Stable', lambda e, total: 1.0)]:
            for epoch in range(phase):
                a = alpha(epoch, phase)
                d_loss_total = g_loss_total = batch_count = 0

                for i in tqdm(range(0, len(dataset), batch_size), desc=f"{phase_name} Epoch {epoch+1}/{phase}"):
                    d_optim.zero_grad()
                    real_samples = dataset[np.random.randint(0, len(dataset), batch_size//2)]
                    x_real = torch.from_numpy(real_samples).float().to(device)
                    x_real = scale_images(x_real, current_size)
                    x_real.requires_grad_()
                    d_real = discriminator(x_real, step, a)
                    loss_real = -torch.mean(d_real)
                    if not torch.isfinite(loss_real):
                        continue
                    r1_penalty = compute_r1_penalty(d_real, x_real)
                    z = torch.randn(batch_size//2, latent_dim, device=device)
                    with torch.no_grad():
                        x_fake = generator(z, step, a)
                    d_fake = discriminator(x_fake.detach(), step, a)
                    loss_fake = torch.mean(d_fake)
                    if not torch.isfinite(loss_fake):
                        continue
                    d_loss = loss_real + loss_fake + r1_weight * r1_penalty
                    if torch.isnan(d_loss):
                        print("[!] NaN detected in D loss. Skipping batch.")
                        continue
                    d_loss.backward()
                    torch.nn.utils.clip_grad_value_(discriminator.parameters(), 0.1)
                    d_optim.step()

                    g_optim.zero_grad()
                    z = torch.randn(batch_size//2, latent_dim, device=device, requires_grad=True)
                    x_fake = generator(z, step, a)
                    g_loss = -torch.mean(discriminator(x_fake, step, a))
                    if step > 1:
                        pl_noise = torch.randn_like(x_fake) / np.sqrt(x_fake.shape[2] * x_fake.shape[3])
                        pl_grads = torch.autograd.grad((x_fake * pl_noise).sum(), inputs=z, create_graph=True, only_inputs=True)[0]
                        pl_length = torch.mean(pl_grads.pow(2).sum(dim=1))
                        g_loss += 0.01 * pl_length
                    if torch.isnan(g_loss):
                        print("[!] NaN detected in G loss. Skipping batch.")
                        continue
                    g_loss.backward()
                    torch.nn.utils.clip_grad_value_(generator.parameters(), 0.1)
                    g_optim.step()
                    ema.update()

                    d_loss_total += d_loss.item()
                    g_loss_total += g_loss.item()
                    batch_count += 1

                if batch_count == 0:
                    print(f"[!] All batches skipped in {phase_name} Step {step} Epoch {epoch+1} due to NaN.")
                    if phase_name == 'Stable':
                        fid_writer.writerow([step, epoch+1, 'NaN'])
                    continue

                avg_d_loss = d_loss_total / batch_count
                avg_g_loss = g_loss_total / batch_count
                all_d_losses.append(avg_d_loss)
                all_g_losses.append(avg_g_loss)

                if phase_name == 'Stable':
                    with torch.no_grad():
                        ema.apply_to(generator)
                        real = scale_images(torch.from_numpy(dataset[np.random.randint(0, len(dataset), 64)]).float().to(device), current_size)
                        z = torch.randn(64, latent_dim, device=device)
                        fake = generator(z, step, a)
                        fid = compute_fid(real, fake)
                        fid_scores.append(fid)
                        fid_writer.writerow([step, epoch+1, fid])
                        print(f"[FID] Step {step} Epoch {epoch+1}: {fid:.2f}")

                metrics_writer.writerow([epoch+1, phase_name, f'{current_size}x{current_size}', avg_d_loss, avg_g_loss, a])
                csv_file.flush()

                with torch.no_grad():
                    z = torch.randn(16, latent_dim, device=device)
                    fake_images = generator(z, step, a)
                    save_image(fake_images, f'training_images/step{step}_{phase_name.lower()}_{epoch+1}.png', nrow=4, normalize=True, value_range=(-1, 1))
                    ema.apply_to(generator)

        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'g_optim': g_optim.state_dict(),
            'd_optim': d_optim.state_dict(),
            'ema': ema.shadow_params
        }, f'model_checkpoints/model_step{step}_final.pth')

    plt.figure(figsize=(10, 5))
    plt.plot(all_d_losses, label='D Loss')
    plt.plot(all_g_losses, label='G Loss')
    plt.plot(fid_scores, label='FID Score')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_metrics/loss_fid_curve_final.png')
    csv_file.close()
    fid_log_file.close()


# ====================== MAIN ======================
if __name__ == "__main__":
    print("Initializing models...")
    generator = ProgressiveGenerator().to(device)
    discriminator = ProgressiveDiscriminator().to(device)

    print("Loading dataset...")
    dataset = load_real_samples('celeba_faces_rtx3060_optimized.npz', max_samples=50000)

    print("Starting training from step 2 stable...")
    train(generator, discriminator, dataset, start_step=2)