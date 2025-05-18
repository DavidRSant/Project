# Comparative Study of Generative Models for Facial Image Synthesis

##  Overview
This project presents a comparative study between two Generative Adversarial Network (GAN) architectures:
- **Progressive Growing GAN (PGGAN)**
- **StyleSwin GAN**

The models were trained to generate synthetic facial images from the CelebA dataset. A detailed analysis is provided on architecture design, training behavior, output quality, and performance metrics such as FID (Fréchet Inception Distance).

---

##  Dataset and Preprocessing
- **Dataset**: [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) – 202,599 images with 40 facial attributes.
- **Preprocessing**:
  - Used **MTCNN** for face detection and cropping.
  - Normalized images to \[-1, 1\] and resized progressively from 4×4 to 128×128.
  - A subset of **50,000 cleaned face images** was used for both models.

---

## Architectures

### PGGAN (Progressive Growing GAN)
- **Reference**: [Karras et al. 2018](https://arxiv.org/abs/1710.10196)
- **Framework**: Implemented in PyTorch based on a ported version of the original Keras code.
- **Key Characteristics**:
  - Starts from a low-resolution (4×4) image and progressively grows the network.
  - New layers are added gradually (fade-in) to improve stability.
  - Generator: uses upsampling + convolutional blocks.
  - Discriminator: uses downsampling blocks.
- **Hyperparameter tuning**:
  - Attempted automated tuning using **Optuna**.
  - Best performance achieved via manual tuning based on the paper.

   ## Generator – `ProgressiveGenerator`
```python
class ProgressiveGenerator(nn.Module):
    def __init__(self, latent_dim=512, depth=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.depth = depth

        # Initial block: linear transformation from latent vector to 4×4 feature map
        self.initial = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.Unflatten(1, (128, 4, 4)),
            PixelNormalization(),  # normalize features to stabilize training
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
                nn.Upsample(scale_factor=2, mode='nearest'),  # double spatial resolution
                Conv2dWS(128, 128, 3, padding=1),
                PixelNormalization(),
                nn.LeakyReLU(0.2),
                Conv2dWS(128, 128, 3, padding=1),
                PixelNormalization(),
                nn.LeakyReLU(0.2)
            )
            self.blocks.append(block)
            self.to_rgb.append(Conv2dWS(128, 3, 1))  # 1x1 conv to RGB output
```

## Discriminator – `ProgressiveDiscriminator`
```python
class ProgressiveDiscriminator(nn.Module):
    def __init__(self, n_blocks=6):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.from_rgb = nn.ModuleList()

        # Final discriminator block: adds minibatch standard deviation, then 1x1 conv to output
        self.final_conv = nn.Sequential(
            MinibatchStdev(),  # adds 1 channel with per-pixel std across minibatch
            Conv2dWS(128 + 1, 1, 1)  # output single-channel real/fake score
        )

        # Add block layers (reverse order)
        self._add_block(128, 128)
        for _ in range(1, n_blocks):
            self._add_block(128, 128)

    def _add_block(self, in_ch, out_ch):
        block = nn.Sequential(
            Conv2dWS(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2),
            Conv2dWS(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2)  # downsample spatial resolution by half
        )
        self.blocks.insert(0, block)
        self.from_rgb.insert(0, Conv2dWS(3, in_ch, 1))  # 1x1 conv to transform RGB to features
```

## Regularization Techniques

<pre>
| Technique                     | Code Implementation                                               |
|------------------------------|-------------------------------------------------------------------|
| Pixel Normalization          | `PixelNormalization` after convolutions in the  generator          |
| Minibatch StdDev             | `MinibatchStdev()` in the final discriminator block               |
| R1 Gradient Penalty          | `compute_r1_penalty()` applied to real images                     |
| Gradient Clipping            | `clip_grad_value_` on optimizer gradients                         |
| Exponential Moving Average   | `ExponentialMovingAverage()` for the generator weights            |
| Fade-in Transition           | Controlled via `alpha` blending RGB outputs during transitions    |
| Path Length Regularization   | Applied when `step > 1` using noise + gradient consistency        |
</pre>


## hyperparameters:
| Parametre           | Value                                |
| ------------------- | ------------------------------------ |
| `latent_dim`        | `512`                                |
| `lr_Generator`      | `0.0005` (Adam)                      |
| `lr_Discriminator`  | `0.00005` (Adam)                     |
| `betas` (Adam)      | `(0.0, 0.99)`                        |
| `n_batch` for step  | `[16, 16, 16, 8, 4, 4]`              |
| `n_epochs` for step | `[5, 8, 8, 10, 10, 10]`              |
| `start_step`        | `0`                                  |
| `ema_decay`         | `0.999`                              |
| `r1_weight`         | `0.01`                               |
| `grad_clip_value`   | `0.1` (used in `.clip_grad_value_`) |











### StyleSwin GAN
- **Reference**: [StyleSwin GitHub (Microsoft)](https://github.com/microsoft/StyleSwin)
- **Framework**: Original PyTorch implementation.
- **Key Characteristics**:
  - Hybrid design with **style modulation** and **Swin Transformer blocks**.
  - Fixed resolution stages (no progressive growth).
  - Generator uses hierarchical attention with shifted windows.
  - Discriminator integrates patch-based contrastive learning.
  - Modulates style at each resolution level.
- **Hyperparameters**:
  - Default settings from the official repo.
  - Modified only `max_iter`, `save_freq`, and `eval_freq` to speed up training.

 




## StyleSwin Generator and Discriminator

```python
# generator.py (StyleSwin)
class Generator(nn.Module):
    def __init__(self, size, style_dim, n_mlp, channel_multiplier=2, ...):
        # Style Encoder: latent input processed through normalization and MLP
        layers = [PixelNorm()]  # normalize latent code
        for _ in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim, ...))  # project style
        self.style = nn.Sequential(*layers)

        # Constant learned 4x4 input tensor
        self.input = ConstantInput(in_channels[0])

        # Main network blocks and ToRGB conversion
        self.layers = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        for i_layer in range(start, end + 1):
            in_channel = in_channels[i_layer - start]
            # Main Swin Transformer block with style injection and optional upsample
            layer = StyleBasicLayer(
                dim=in_channel,
                input_resolution=(2 ** i_layer, 2 ** i_layer),
                ...
            )
            self.layers.append(layer)

            # RGB conversion after each resolution stage
            out_dim = in_channels[i_layer - start + 1] if (i_layer < end) else in_channels[i_layer - start]
            to_rgb = ToRGB(out_dim, upsample=(i_layer < end), resolution=(2 ** i_layer))
            self.to_rgbs.append(to_rgb)
```

```python
# discriminator.py (StyleSwin)
class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], sn=False, ssd=False):
        super().__init__()

        # Channel configuration for each resolution level
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        # Wavelet-based preprocessing
        self.dwt = HaarTransform(3)

        # Convolutional feature extractors
        self.from_rgbs = nn.ModuleList()
        self.convs = nn.ModuleList()

        # Build discriminator blocks from high to low resolution
        log_size = int(math.log(size, 2)) - 1
        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            self.from_rgbs.append(FromRGB(in_channel, downsample=i != log_size, sn=sn))
            self.convs.append(ConvBlock(in_channel, out_channel, blur_kernel, sn=sn))
            in_channel = out_channel

        # Final processing layers
        self.from_rgbs.append(FromRGB(channels[4], sn=sn))
        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, sn=sn)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )
```


## Regularization Techniques Used (StyleSwin)
<pre>
| Technique                     | Code Implementation                                                       |
|------------------------------|----------------------------------------------------------------------------|
| Dropout                      | Used in Swin Transformer blocks and attention modules                      |
| Attention Masking            | Shifted window-based self-attention ensures locality                       |
| Weight Decay                 | Applied via AdamW optimizer (standard in transformer training)             |
| LayerNorm                    | Applied before MLP and attention sub-blocks                                |
| Data Augmentation            | Includes resize, flip, crop, color jitter (via default training pipeline) |
</pre>

## Hyperparameters Used (StyleSwin)
<pre>
| Parameter              | Value                    |
|------------------------|--------------------------|
| Optimizer              | AdamW                    |
| Learning Rate          | 2e-4                     |
| Weight Decay           | 0.05                     |
| Batch Size             | 4                        |
| Training Resolution    | 128×128                  |
| Iterations             | 80k                      |
| Save Freq / Eval Freq  | 10k                      |
</pre>

**Generator Overview:**
- Starts from a fixed 4×4 learnable tensor.
- Style vector modulates blocks using MLP (EqualLinear).
- Attention blocks (StyleBasicLayer) apply Swin-style self-attention.
- Bilinear upsampling and positional encoding preserve spatial structure.
- ToRGB maps features to RGB space.

**Discriminator Overview:**
- Input processed with HaarTransform for hierarchical frequency decomposition.
- Uses convolution and SwinDiscriminatorBlock layers.
- Minibatch statistics improve discrimination.
- Final EqualLinear layer outputs real/fake prediction.



---

## Hardware Setup
- CPU: AMD Ryzen 5 5500
- RAM: 32 GB DDR4
- GPU: NVIDIA RTX 3060 (12 GB VRAM)

---

## Results Summary
| Metric        | PGGAN (32×32, step 3) | StyleSwin (128×128) |
|---------------|------------------------|---------------------|
| Max Resolution| 32×32 (step 3 only)    | 128×128             |
| Training Stability | Low (failed at step 4, 64×64) | High stability       |
| FID Score     | 330–599                | ~38                 |
| Visual Quality| Noticeable artifacts   | High fidelity        |

> **Note**: PGGAN failed to transition properly to step 4 (64×64), resulting in invalid or non-evaluable outputs (null FID). Metrics reported for PGGAN correspond to step 3 (32×32).

![001310](https://github.com/user-attachments/assets/9c57bc1d-39c8-4333-a2d7-5a63466cdebc)
![001308](https://github.com/user-attachments/assets/80ef4e95-581f-4116-8f0d-229d9a5e23ca)
![001307](https://github.com/user-attachments/assets/fe663638-dde2-491e-befb-0456e3c457ee)
![001306](https://github.com/user-attachments/assets/18cb0b23-c8f6-45b3-b216-03a19c12e02d)
![001305](https://github.com/user-attachments/assets/285c5412-6627-4fa1-858d-5cb1cc4b936a)
![001304](https://github.com/user-attachments/assets/02e9e816-8adf-4428-b802-fc3fd65ac9d4)
![001303](https://github.com/user-attachments/assets/27bdb241-0c93-498e-80ea-3984ff97b6a3)
![001240](https://github.com/user-attachments/assets/70976c6f-b2b8-4f7d-8e13-92180128e1aa)
![001239](https://github.com/user-attachments/assets/4ea8ac8b-74d5-4fb1-979d-96276b60c0ad)
![001238](https://github.com/user-attachments/assets/83bfb7e7-7b30-4e80-8773-5fb937c0c764)
![001236](https://github.com/user-attachments/assets/8eb19a3b-406d-46e2-bd53-b9376f7b27ec)
![001299](https://github.com/user-attachments/assets/66166865-72b7-4364-9a98-464f652a052b)
![001298](https://github.com/user-attachments/assets/9ba62963-be0e-4cf0-85eb-5abebad41888)
![001297](https://github.com/user-attachments/assets/69c0e66b-4d1d-4b03-ab92-8a18abe021cf)
![001296](https://github.com/user-attachments/assets/8f6e191c-f310-4b04-8d40-394739e518e7)
![000014](https://github.com/user-attachments/assets/b0ee460f-6390-44bb-84f6-41c2ac65b170)
![001294](https://github.com/user-attachments/assets/f37c0bd7-5583-4fd0-ad5b-5f338369a52e)
![001293](https://github.com/user-attachments/assets/2cff311d-4a2e-4092-b27f-1b53ea6040c5)
![001292](https://github.com/user-attachments/assets/ba3271c8-11fe-423d-92b9-f0fa389e4c15)
![001291](https://github.com/user-attachments/assets/9252ba1d-49b5-47ff-be42-231e5c0d0585)
![001290](https://github.com/user-attachments/assets/6881fb3a-cc2c-44b6-99a1-5a83594a16c3)
![001289](https://github.com/user-attachments/assets/2e434c1f-9dd1-4c4a-94c2-8bbbc91747fb)
![001288](https://github.com/user-attachments/assets/4ea3bfda-e18a-4ac8-893e-e08e39ffe88b)
![001287](https://github.com/user-attachments/assets/54d58002-b6f9-4eb1-b103-51c5c6761120)
![001286](https://github.com/user-attachments/assets/a9df5e42-1322-4465-98aa-6e131a49d5dc)
![001285](https://github.com/user-attachments/assets/164706bb-bf0a-4145-974d-80387a326c8e)
![001284](https://github.com/user-attachments/assets/96d2a4fd-830f-4939-9867-45df8f3f576c)
![001283](https://github.com/user-attachments/assets/6d5ce7a7-2e27-4eb7-bd60-4843154388f2)
![001282](https://github.com/user-attachments/assets/df6e42b3-6ab6-4692-a88f-79a4d9eb1484)
![001281](https://github.com/user-attachments/assets/4648b09c-ec30-465e-a430-67dad1af0532)
![001280](https://github.com/user-attachments/assets/21e74bf3-67ed-4666-88e3-1b820eb49566)
![001279](https://github.com/user-attachments/assets/0a9f8054-5fa5-4600-96e5-ca71f0f757f9)
![001278](https://github.com/user-attachments/assets/541e24d1-4382-44cd-98c7-0489a8f48051)
![001277](https://github.com/user-attachments/assets/11c9ef9f-7f50-4c61-8d46-9dbc5994fcd1)
![001276](https://github.com/user-attachments/assets/c75bb420-17b7-4a6b-85d4-e07111fc3b3a)
![001275](https://github.com/user-attachments/assets/1d97bc01-97f3-4e63-8d56-12f3f5cb66c6)
![001274](https://github.com/user-attachments/assets/83f11a66-ddf9-4dbc-8d08-f76de50702ae)
![001273](https://github.com/user-attachments/assets/1b2e12fa-e7dc-4c6c-80f9-ea8fb8a2bd0a)
![001272](https://github.com/user-attachments/assets/fe19cd3e-862d-4544-89f8-7154f158aa4c)
![001271](https://github.com/user-attachments/assets/0540fae8-9f63-4cbd-89c0-fc9374c0767b)
![001270](https://github.com/user-attachments/assets/60e1aed2-eb4d-42be-8f12-6ab73f1e6074)
![001269](https://github.com/user-attachments/assets/3f610000-16aa-44b3-bf84-2074c0b052b9)
![001268](https://github.com/user-attachments/assets/52531a6b-e30e-41bc-9cf4-1cb795e18be1)
![001331](https://github.com/user-attachments/assets/11ed6246-e019-4b57-9250-28d6970824eb)
![001330](https://github.com/user-attachments/assets/cc3d9ec4-d35e-4505-a3ac-58de210f092c)
![001329](https://github.com/user-attachments/assets/877c7597-e861-437e-9cff-31d19365d537)
![001328](https://github.com/user-attachments/assets/d33ed605-8e79-4625-a3b4-d60a7575501d)
![001327](https://github.com/user-attachments/assets/45fe0205-4642-4d68-8a2f-ef3e7be7f960)
![001326](https://github.com/user-attachments/assets/2ea28a7e-20f3-4af2-a374-122342cf818a)
![001325](https://github.com/user-attachments/assets/13500834-5ca7-417e-b393-cb447f4528e8)
![001324](https://github.com/user-attachments/assets/0ba13e63-a1d4-425e-a352-76a99c7a7cc2)
![001323](https://github.com/user-attachments/assets/20fa66ef-27c4-4175-aa09-55a599f76d57)
![001322](https://github.com/user-attachments/assets/5174d85a-9c8d-42b4-b634-5613a51c8307)
![001321](https://github.com/user-attachments/assets/393c6095-e4a7-40ff-a26b-adcea416c0f7)
![001320](https://github.com/user-attachments/assets/381492c7-8557-433d-9f88-ce87e4a34494)
![001319](https://github.com/user-attachments/assets/b8b9e834-3a26-4737-8ffe-b0393cd61547)
![001318](https://github.com/user-attachments/assets/fc5fdc05-50e1-4b17-a260-c2d76adaf3f2)
![001317](https://github.com/user-attachments/assets/882f9c77-4b45-4594-9f17-6c41dacafca3)
![001316](https://github.com/user-attachments/assets/8b0df304-ad3d-4f34-b501-29e673a96c04)
![001315](https://github.com/user-attachments/assets/e26f364d-2951-4f93-819a-5a5a85538daf)
![001314](https://github.com/user-attachments/assets/6889f95c-7c81-423a-b05f-59511a321ea7)
![001313](https://github.com/user-attachments/assets/17e334f4-38e2-43e6-a3ec-1963f8b8f2c3)
![001312](https://github.com/user-attachments/assets/ab8df1b4-53b5-455e-aee9-7d8cd99e33bf)
![001311](https://github.com/user-attachments/assets/4ec069e2-87ad-4e43-8017-55c414074a24)
![001310](https://github.com/user-attachments/assets/9bb9329e-e5bf-4f3e-8870-fe746d9292c6)
![001309](https://github.com/user-attachments/assets/01521535-d561-489b-86c6-e72e51f10a16)
![001243](https://github.com/user-attachments/assets/5dbd96a1-df60-44e1-bf6c-bddd6ddd9024)
![001242](https://github.com/user-attachments/assets/8da80c62-cf3b-4338-9e7a-c8a242926b5c)
![001241](https://github.com/user-attachments/assets/8f916f7b-ec0c-4f19-aba9-7636f0399cc0)
![001244](https://github.com/user-attachments/assets/b0f87f77-c1d5-4893-a541-70dbd0cf87b1)

examples of images generated by styleswin with max_iter=80k(default=800k).



![step3_stable_15](https://github.com/user-attachments/assets/bd136458-e3dc-488f-ab43-7860c68abb82)

examples of images generated by Pggan 32x32 resolution.






---

## Observations
- PGGAN struggled with transition phases, requiring careful tuning. Training instability was especially pronounced during the transition from step 3 to step 4 (64×64), which could potentially be mitigated through improved fade-in scheduling or hyperparameter tuning. However, due to persistent degradation at this stage, the model was replaced by a more modern and stable architecture.
- StyleSwin performed consistently across training steps with better global coherence due to attention mechanisms.

---

## Future Work
- Extend output resolution to 256×256 or higher.
- Explore **fairness-aware training** to address dataset bias.
- Investigate improved preprocessing pipelines (e.g., facial alignment).
- Incorporate **StyleGAN3** or **Diffusion models** for baseline comparison.
- Optimize training with more robust hyperparameter search.
- Explore the generalization capability of the StyleSwin architecture by applying it to other domains, such as medical image synthesis.


---

## License
MIT License

---

## Author
David Rocha de Santana  
Department of Computer Science  
Universidade Federal Fluminense  
[davidrs@id.uff.br](mailto:davidrs@id.uff.br)
