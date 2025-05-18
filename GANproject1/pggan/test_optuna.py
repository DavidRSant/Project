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
from torch.nn.functional import adaptive_avg_pool2d
import optuna
from optuna.samplers import TPESampler
from joblib import Parallel, delayed
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import multiprocessing
from scipy import linalg  # <-- FIXED MISSING IMPORT

# Enable fast GPU kernels
cudnn.benchmark = True

# Use mixed precision if available (requires torch >= 1.6)
use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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
    X = data['faces']
    X = np.array(X[:max_samples], dtype=np.float32)  # force lazy eval after slicing
    print(f"Dataset range before normalization: {X.min()} to {X.max()}")
    return (X.transpose(0, 3, 1, 2) * 2) - 1

def setup_directories():
    os.makedirs('training_images', exist_ok=True)
    os.makedirs('training_metrics', exist_ok=True)
    os.makedirs('model_checkpoints', exist_ok=True)

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

    # Proteção contra NaN
    if not np.isfinite(act1).all() or not np.isfinite(act2).all():
        print("[!] Activation contains NaNs or infs")
        return float("inf")

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    if not (np.isfinite(mu1).all() and np.isfinite(sigma1).all() and
            np.isfinite(mu2).all() and np.isfinite(sigma2).all()):
        print("[!] Covariance or mean contains NaNs")
        return float("inf")

    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real
    if not np.isfinite(covmean).all():
        print("[!] sqrtm produced invalid values")
        return float("inf")

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# ====================== DATASET ======================
class CelebADataset(Dataset):
    def __init__(self, npz_path, max_samples):
        with np.load(npz_path, mmap_mode='r') as data:
            self.images = data['faces'][:max_samples]
        self.images = (self.images.transpose(0, 3, 1, 2).astype(np.float32) * 2) - 1
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

# Detect CPU count for DataLoader optimization
NUM_WORKERS = multiprocessing.cpu_count()

# ====================== DATA LOADER ======================
def get_dataloader(npz_path, batch_size, max_samples):
    dataset = CelebADataset(npz_path, max_samples=max_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

# ====================== LOAD DATA FOR TRAIN ======================
def load_real_samples(npz_path, batch_size, max_samples):
    return get_dataloader(npz_path, batch_size=batch_size, max_samples=max_samples)

# ====================== TRAIN FUNCTION ======================
def train(hparams, dataloader):
    best_fid = float('inf')
    overall_fid_log = []
    local_scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for step in range(hparams['start_step'], 6):
        generator = ProgressiveGenerator(latent_dim=hparams['latent_dim']).to(device)
        discriminator = ProgressiveDiscriminator().to(device)
        g_optim = optim.Adam(generator.parameters(), lr=hparams['lr_g'], betas=(hparams['beta1'], hparams['beta2']))
        d_optim = optim.Adam(discriminator.parameters(), lr=hparams['lr_d'], betas=(hparams['beta1'], hparams['beta2']))
        ema = ExponentialMovingAverage(generator.parameters())

        patience_counter = 0
        fid_log = []
        fade_epochs = hparams['fade_epochs']
        stable_epochs = hparams['stable_epochs']
        total_epochs = fade_epochs + stable_epochs
        size = 4 * (2 ** step)

        for epoch in range(total_epochs):
            alpha = min(1.0, epoch / max(1, fade_epochs - 1))
            generator.train()
            discriminator.train()

            for real_images in dataloader:
                real_images = real_images.to(device, non_blocking=True)
                real_images = scale_images(real_images, size)

                z = torch.randn(real_images.size(0), hparams['latent_dim'], device=device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    fake_images = generator(z, step, alpha)
                    d_real = discriminator(real_images, step, alpha)
                    d_fake = discriminator(fake_images.detach(), step, alpha)
                    d_loss = -torch.mean(d_real) + torch.mean(d_fake)

                d_optim.zero_grad()
                if torch.isfinite(d_loss):
                    local_scaler.scale(d_loss).backward()
                    local_scaler.step(d_optim)
                    local_scaler.update()

                z = torch.randn(real_images.size(0), hparams['latent_dim'], device=device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    fake_images = generator(z, step, alpha)
                    g_loss = -torch.mean(discriminator(fake_images, step, alpha))

                g_optim.zero_grad()
                if torch.isfinite(g_loss):
                    local_scaler.scale(g_loss).backward()
                    local_scaler.step(g_optim)
                    local_scaler.update()
                ema.update()

            if epoch >= fade_epochs:  # Only compute FID during Stable phase
                with torch.no_grad():
                    ema.apply_to(generator)
                    z = torch.randn(64, hparams['latent_dim'], device=device)
                    fake = generator(z, step, alpha)
                    real_subset = next(iter(dataloader))[:64].to(device)
                    real = scale_images(real_subset, size)
                    fid = compute_fid(real, fake)
                    fid_log.append((step, epoch + 1, fid))
                    print(f"[Trial FID] Step {step} Epoch {epoch + 1}: {fid:.2f}")
                    if fid < best_fid:
                        best_fid = fid
                        patience_counter = 0
                        torch.save(generator.state_dict(), f'model_checkpoints/best_generator_step{step}.pth')
                    else:
                        patience_counter += 1
                    if patience_counter >= hparams['early_stopping_patience']:
                        break

        overall_fid_log.extend(fid_log)
        if patience_counter >= hparams['early_stopping_patience']:
            break

    with open('training_metrics/fid_scores_trials.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for row in overall_fid_log:
            writer.writerow(row)

    return best_fid



# ====================== OPTUNA OBJECTIVE UPDATED ======================
def objective(trial):
    hparams = {
        "lr_g": trial.suggest_float("lr_g", 1e-5, 5e-3, log=True),
        "lr_d": trial.suggest_float("lr_d", 1e-5, 5e-3, log=True),
        "beta1": trial.suggest_float("beta1", 0.0, 0.5),
        "beta2": trial.suggest_float("beta2", 0.8, 0.999),
        "r1_weight": trial.suggest_float("r1_weight", 1e-4, 1e-1, log=True),
        "latent_dim": trial.suggest_categorical("latent_dim", [128, 256, 512]),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
        "fade_epochs": trial.suggest_int("fade_epochs", 5, 20),
        "stable_epochs": trial.suggest_int("stable_epochs", 5, 20),
        "max_samples": 20000,
        "start_step": 2,
        "early_stopping_patience": 5
    }
    dataloader = load_real_samples('celeba_faces_rtx3060_optimized.npz', hparams['batch_size'], hparams['max_samples'])
    fid = train(hparams, dataloader)
    return fid



# ====================== MAIN ======================
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", sampler=TPESampler())
    study.optimize(objective, n_trials=20, n_jobs=2)

    print("Best hyperparameters:")
    print(study.best_trial.params)
    print("Top 5 FIDs from completed trials:")
    top_trials = sorted([t for t in study.trials if t.value is not None], key=lambda x: x.value)[:5]
    for t in top_trials:
        print(f"Trial {t.number} - FID: {t.value:.2f}")
