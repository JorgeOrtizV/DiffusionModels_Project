import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler, DataLoader
import torchvision
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter, zoom, mean as ndimage_mean
from skimage.transform import resize
import os
from torchvision.utils import save_image
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(1, mid_channels)
        self.act = nn.GELU()  ## Try Relu, leakyReLU
        self.conv2 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.residual = residual

    def forward(self, x):
        x2 = self.conv1(x)
        x2 = self.norm1(x2)
        x2 = self.act(x2)
        x2 = self.conv2(x2)
        x2 = self.norm2(x2)
        if self.residual:
            return self.act(x + x2)
        else:
            return x2


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxPool = nn.MaxPool2d(2)
        self.doubleConv1 = DoubleConv(in_channels, in_channels, residual=True)
        self.doubleConv2 = DoubleConv(in_channels, out_channels)

        self.act = nn.SiLU()
        self.linear = nn.Linear(emb_dim, out_channels)

    def forward(self, x, t):
        x = self.maxPool(x)
        x = self.doubleConv1(x)
        x = self.doubleConv2(x)
        # print(x.size())
        emb = self.act(t)
        emb = self.linear(emb)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        # print(emb.size())

        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        # self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up = nn.ConvTranspose2d(
            in_channels // 2, in_channels // 2, kernel_size=2, stride=2
        )
        self.doubleConv1 = DoubleConv(in_channels, in_channels, residual=True)
        self.doubleConv2 = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.act = nn.SiLU()
        self.linear = nn.Linear(emb_dim, out_channels)

    def forward(self, x, skip_x, t):

        # print("X:",x.size())
        x = self.up(x)
        # print("X:",x.size())
        if x.shape[-2:] != skip_x.shape[-2:]:
            x = nn.functional.interpolate(
                x, size=skip_x.shape[-2:], mode="bilinear", align_corners=True
            )
            # print(x.size())
        x = torch.cat([skip_x, x], dim=1)
        x = self.doubleConv1(x)
        x = self.doubleConv2(x)
        emb = self.act(t)
        emb = self.linear(emb)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.linear = nn.Linear(channels, channels)
        self.act = nn.GELU()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        x = self.ln(attention_value)
        x = self.linear(x)
        x = self.act(x)
        x = self.linear(x)
        attention_value = x + attention_value

        return attention_value.permute(0, 2, 1).view(b, c, h, w)


class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, device=device):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 14)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 7)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 4)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 7)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 14)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 28)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    # Sinosoidal encoding - further read
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )

        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        # Bottleneck
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        # Decoder
        # print(x4.size())
        # print(x3.size())
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        return self.outc(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_channels=4):
        super().__init__()
        # Input: [batch_size, 1, 28, 28] -> Output: [batch_size, 4, 8, 8]
        self.conv1 = nn.Conv2d(
            in_channels, 32, kernel_size=4, stride=2, padding=1
        )  # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=4, stride=2, padding=2
        )  # 14x14 -> 8x8
        self.conv3 = nn.Conv2d(
            64, latent_channels, kernel_size=3, padding=1
        )  # 8x8 -> 8x8

        self.norm1 = nn.GroupNorm(8, 32)
        self.norm2 = nn.GroupNorm(8, 64)
        self.norm3 = nn.GroupNorm(1, latent_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.norm3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_channels=4, out_channels=1):
        super().__init__()
        # Decoder architecture
        # Input: [batch_size, 4, 8, 8] -> Output: [batch_size, 1, 28, 28]
        self.conv1 = nn.Conv2d(latent_channels, 64, kernel_size=3, padding=1)  # 8x8
        self.conv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=2
        )  # 14x14
        self.conv3 = nn.ConvTranspose2d(
            32, out_channels, kernel_size=4, stride=2, padding=1
        )  # 28x28

        self.norm1 = nn.GroupNorm(8, 64)
        self.norm2 = nn.GroupNorm(8, 32)
        self.act = nn.SiLU()

    def forward(self, x):
        # First convolution maintains spatial dimensions
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        # print("X_1:",x.size())

        # First upsample: 8x8 -> 16x16
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        # print("X_2:",x.size())

        # Second upsample: 16x16 -> 28x28
        x = self.conv3(x)
        # print("X_3:",x.size())
        return torch.tanh(x)  # normalize output to [-1, 1]


class LatentDiffusionModel(nn.Module):
    def __init__(self, encoder, decoder, diffusion_unet, diffusion):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.unet = diffusion_unet
        self.diffusion = (
            diffusion  # cold diff or normal diff based on the instance sent
        )

    def freeze_autoencoder(self):
        """Freeze encoder and decoder weights"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def unfreeze_autoencoder(self):
        """Unfreeze encoder and decoder weights"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True

    def forward(self, x, t):

        z = self.encoder(x)  # encode to latent
        # plot_images(z)
        z_noised = self.diffusion.degradation(
            z, t
        )  # apply diffusion noise in latent space
        # plot_images(z_noised)
        z_denoised = self.unet(z_noised, t)
        # plot_images(z_denoised)
        x_recon = self.decoder(z_denoised)

        return x_recon

    def encode(self, x):
        """Encode images to latent space"""
        return self.encoder(x)

    def decode(self, z):
        """Decode from latent space to image space"""
        return self.decoder(z)

    def autoencoder_forward(self, x):
        """Forward pass through just the autoencoder"""
        z = self.encoder(x)
        return self.decoder(z)


class Diffusion:
    def __init__(
        self, steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device=device
    ):
        self.steps = steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)  # cumulative product.

    def prepare_noise_schedule(self, mode="linear"):
        if mode == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.steps)
        if mode == "cos":
            # TODO : open ai cos schedule
            pass

    def noise_images(self, x, t):
        # Generate X_t in a single step as described in the paper
        # x_t = sqrt(alpha_hat)*x_0 + sqrt(1-alpha_hat)*e
        e = torch.randn_like(x)
        x_t = (
            x * torch.sqrt(self.alpha_hat[t])[:, None, None, None]
            + torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None] * e
        )  # ?
        return x_t, e

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.steps, size=(n,))

    def sample(self, model, n):
        model.eval()
        # Algo 2 - Sampling
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (
                    1
                    / torch.sqrt(alpha)
                    * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise)
                    + torch.sqrt(beta) * noise
                )
            model.train()
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
            return x

    def degradation(self, x, t):
        # Returns noised image x_t and noise e
        x_t, e = self.noise_images(x, t)
        return x_t


class coldDiff:
    def __init__(
        self,
        steps=300,
        size=28,
        loss_type="L1",
        degradation_type="blur",
        device=None,
        blur_sigma=0.33,
    ):

        self.steps = steps
        self.img_size = size
        self.loss_type = loss_type
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.blur_sigma = blur_sigma

        self.beta_schedule = self.get_beta_schedule(self.steps)
        self.alpha = 1.0 - self.beta_schedule
        alphas = torch.cat(
            [torch.tensor([1.0]), self.alpha]
        )  # Add a leading 1.0 to the alphas tensor
        self.alpha_hat = torch.cumprod(alphas, dim=0).to(device)

        # Degradation type can be 'blur', 'pixellate', 'inpainting' or 'snow'
        if degradation_type == "blur":
            self.degradation = self.blur
        # elif degradation_type == 'pixellate':
        #    self.degradation = self.pixellate
        elif degradation_type == "inpainting":
            self.degradation = self.inpainting
        elif degradation_type == "snow":
            self.degradation = self.snow
        elif degradation_type == "gaussian":
            self.degradation = self.gaussian_noise
        else:
            raise ValueError(
                "Invalid degradation type. Choose from 'blur', 'pixellate', 'inpainting' or 'snow'."
            )

    # SAMPLING FUNCTIONS
    def sample_timesteps(self, n):
        return torch.randint(
            low=1, high=self.steps, size=(n,)
        )  # Should this be self.steps+1?

    def sample(
        self,
        model,
        batch_size,
        initial_image="real_degraded",
        data_loader=None,
        gmm=None,
    ):
        output_shape = (batch_size, 1, 28, 28)
        t = self.steps
        with torch.no_grad():
            model.eval()
            # x_prev = x_t
            # black image
            if initial_image == "black":
                x_prev = torch.zeros(output_shape).to(device)

            if initial_image == "random":
                # random and degraded
                dummy_x0 = torch.randn(output_shape).to(
                    device
                )  # Or: real_batch_from_train_loader
                timestep_tensor = torch.tensor([self.steps] * batch_size).to(device)
                x_prev = self.degradation(dummy_x0, timestep_tensor)
            if initial_image == "real_degraded":
                if data_loader is None:
                    raise ValueError(
                        "data_loader is not provided. Please provide a DataLoader for sampling."
                    )
                real_batch, _ = next(iter(data_loader))  # get real MNIST digits
                real_batch = real_batch.to(device)
                real_batch = real_batch[:batch_size]  # in case batch size mismatch
                timestep_tensor = torch.tensor([self.steps] * real_batch.shape[0]).to(
                    device
                )
                x_prev = self.degradation(real_batch, timestep_tensor)

            if initial_image == "gmm":
                # Gaussian Mixture Model
                if gmm is None:
                    raise ValueError(
                        "GMM is not provided. Please provide a GMM model for sampling."
                    )
                # Sample from the GMM
                gmm_samples, _ = gmm.sample(batch_size)  # Sample from the GMM
                gmm_samples = gmm_samples.astype(np.float32)
                gmm_samples = torch.from_numpy(gmm_samples)  # Convert to tensor
                gmm_samples = gmm_samples.view(
                    batch_size, 1, self.img_size, self.img_size
                )  # Reshape to the desired output shape
                gmm_samples = gmm_samples.to(device)  # Move to the appropriate device
                # Set the initial image to the GMM samples
                x_prev = gmm_samples

            for s in range(t, 0, -1):
                s_ = (torch.ones(batch_size) * s).long().to(device)
                pred_x0 = model(x_prev, s_)
                x_prev = (
                    x_prev
                    - self.degradation(pred_x0, s_)
                    + self.degradation(pred_x0, s_ - 1)
                )
        return x_prev

    def sample_ddpm(self, model, batch_size):
        model.eval()
        # Algo 2 - Sampling
        with torch.no_grad():
            x = torch.randn((batch_size, 1, self.img_size, self.img_size)).to(
                self.device
            )
            for i in tqdm(reversed(range(1, self.steps)), position=0):
                t = (torch.ones(batch_size) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta_schedule[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (
                    1
                    / torch.sqrt(alpha)
                    * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise)
                    + torch.sqrt(beta) * noise
                )
            model.train()
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
            return x

    # Normal gaussian noise degradation
    def gaussian_noise(self, x0, t):
        """if len(t.shape) == 1:
            t = t[:, None, None, None]

        sqrt_alpha_bar = self.alpha_bar[t.squeeze().long().clamp(max=self.steps)].sqrt().to(x0.device)[:, None, None, None]
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bar[t.squeeze().long().clamp(max=self.steps)]).sqrt().to(x0.device)[:, None, None, None]

        noise = torch.randn_like(x0)

        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        """

        e = torch.randn_like(x0)
        x_t = (
            x0 * torch.sqrt(self.alpha_hat[t])[:, None, None, None]
            + torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None] * e
        )
        return x_t

    # DEGRADATION FUNCTIONS
    def blur(self, x0, t):
        sigma_t = torch.sqrt(t * self.blur_sigma**2).detach().cpu().numpy()
        x0_cpu = x0.detach().cpu().numpy()
        blurred_imgs = [
            gaussian_filter(x0_cpu[i], sigma=(0, sigma_t[i], sigma_t[i]))
            for i in range(len(sigma_t))
        ]
        return torch.from_numpy(np.stack(blurred_imgs)).to(device)
        # return GaussianBlur(5, sigma_t)

    def inpainting(self, x0, t, base_variance=1):  # Base variance is beta in the paper
        w, h = x0.shape[2], x0.shape[3]
        center_x, center_y = np.random.randint(
            0, w, size=x0.shape[0]
        ), np.random.randint(0, h, x0.shape[0])
        variance = base_variance + 0.5 * t.detach().cpu().numpy()
        # 2d gaussian curve with center at rand_x,rand_y and peak value = 1, discretized

        x0 = x0.detach().cpu().numpy()
        gaussian_mask = np.zeros_like(x0, dtype=float)

        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        gaussian = np.stack(
            [
                np.exp(
                    -((x - center_x[i]) ** 2 + (y - center_y[i]) ** 2)
                    / (2 * variance[i])
                )
                for i in range(len(x0))
            ]
        )
        gaussian = 1 - gaussian

        # Normalize the gaussian mask
        gaussian = gaussian / np.max(gaussian, axis=(1, 2), keepdims=True)

        # Add extra channel dimension as 2nd dimension
        gaussian = gaussian[:, None, :, :]

        # Apply the mask to the image
        inpainted_imgs = x0 * gaussian

        return torch.from_numpy(inpainted_imgs).float().to(device)

    ## Scheduling
    def get_beta_schedule(
        self, timesteps, start=1e-4, end=0.02, schedule_type="linear"
    ):
        if schedule_type == "linear":
            return torch.linspace(start, end, timesteps)
        elif schedule_type == "cosine":
            steps = torch.arange(timesteps + 1, dtype=torch.float32)
            f = (
                lambda t: torch.cos(((t / timesteps) + 0.008) / 1.008 * torch.pi / 2)
                ** 2
            )
            alphas_bar = f(steps) / f(torch.tensor(0.0))
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            return torch.clip(betas, 1e-5, 0.999)
        else:
            raise ValueError("Unknown schedule type")


#####################################################
# Initialize model
parser = argparse.ArgumentParser(description="Latent Diffusion Model")
parser.add_argument("--lc", type=int, default=4, help="Number of latent channels")
parser.add_argument(
    "--img_size", type=int, default=28, help="Size of the input images (default: 28)"
)
parser.add_argument(
    "--steps", type=int, default=100, help="Number of diffusion steps (default: 100)"
)
parser.add_argument(
    "--cold_diff", type=bool, default=False, help="Use cold diffusion (default: False)"
)
parser.add_argument(
    "--blur_sigma",
    type=float,
    default=0.33,
    help="Blur sigma for cold diffusion (default: 0.33)",
)

args = parser.parse_args()

latent_channels = args.lc


encoder = Encoder(in_channels=1, latent_channels=latent_channels).to(device)
decoder = Decoder(latent_channels=latent_channels, out_channels=1).to(device)
unet = UNet(c_in=latent_channels, c_out=latent_channels).to(device)
if args.cold_diff:
    diffusion = coldDiff(
        steps=args.steps,
        size=args.img_size,
        degradation_type="blur",
        device=device,
        blur_sigma=args.blur_sigma,
    )
else:
    diffusion = Diffusion()
ldm = LatentDiffusionModel(encoder, decoder, unet, diffusion).to(device)

# Test the dimensions
with torch.no_grad():
    test_input = torch.randn(2, 1, 28, 28).to(device)
    latent = encoder(test_input)
    print("Latent shape:", latent.shape)
    unet_out = unet(latent, torch.ones(2).to(device))
    print("UNet output shape:", unet_out.shape)  # Should also be [2, 4, 7, 7]
    recon = decoder(unet_out)
    print("Recon shape:", recon.shape)

transforms = torchvision.transforms.Compose(
    [
        # torchvision.transforms.Resize(80),
        # torchvision.transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        torchvision.transforms.Resize((28, 28)),  # Resize to 28x28 for MNIST
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)

np.random.seed(32)


def split_indices(n, val_pct):
    n_val = int(val_pct * n)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]


autoencoder_epochs = 20
diffusion_epochs = 100

batch_size = 32
image_size = args.img_size  # 28 for MNIST, 32 for CIFAR10
learning_rate = 1e-4
loss_type = "L1"  # 'L1', 'L2', 'SmoothL1', 'Huber'
sampling_type = "gmm"  # 'random', 'black', 'real_degraded', 'gmm'

# DEGRADATION TYPE
degradation_type = "blur"  # 'blur', 'pixellate', 'inpainting', 'snow', 'gaussian'

# Steps for diffusion process - changes based on the function used
# For pixellate, steps = 4 (MNIST) or 6 (CIFAR10)
steps = args.steps  # 1000 for MNIST, 250 for CIFAR10

FASHION_MNIST = False

# Import dataset
if FASHION_MNIST:
    dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=transforms, download=True
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=transforms, download=True
    )
else:
    dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transforms, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transforms, download=True
    )

# LIMIT THE DATASET
LIMIT = True
if LIMIT:
    num_samples = 6016  # FOR LIMIT
    indices = np.random.permutation(len(dataset))[:num_samples]
    dataset = torch.utils.data.Subset(dataset, indices)
    test_indices = np.random.permutation(len(test_dataset))[: num_samples // 6]
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    print("LIMITED DATASET")
    print(len(dataset))
    print(len(test_dataset))

else:
    print("FULL DATASET")
    print(len(dataset))
    print(len(test_dataset))
# dataset = torch.utils.data.Subset(dataset, range(1000))
# test_dataset = torch.utils.data.Subset(test_dataset, range(1000))


train_indices, val_indices = split_indices(len(dataset), 0.2)
train_sampler = SubsetRandomSampler(train_indices)
train_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
val_sampler = SubsetRandomSampler(val_indices)
val_loader = DataLoader(dataset, batch_size, sampler=val_sampler)

model = UNet().to(device)


autoencoder_optimizer = torch.optim.Adam(
    list(ldm.encoder.parameters()) + list(ldm.decoder.parameters()), lr=1e-4
)

diffusion_optimizer = torch.optim.Adam(ldm.unet.parameters(), lr=1e-4)

# Create optimizer for diffusion model only

autoencoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    autoencoder_optimizer, factor=0.5, patience=5
)
diffusion_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    diffusion_optimizer, factor=0.5, patience=5, min_lr=1e-6
)


if loss_type == "L1":
    loss_fn = nn.L1Loss()
elif loss_type == "L2":
    loss_fn = nn.MSELoss()
elif loss_type == "SmoothL1":
    loss_fn = nn.SmoothL1Loss()
elif loss_type == "Huber":
    loss_fn = nn.HuberLoss()
elif loss_type == "BCE":
    loss_fn = nn.BCEWithLogitsLoss()
else:
    raise ValueError(
        "Invalid loss type. Choose from 'L1', 'L2', 'SmoothL1', 'Huber' or 'BCE'."
    )


# Loss for autoencoder
reconstruction_loss = nn.MSELoss()

length = len(train_loader)
print(length)

train_losses = []
val_losses = []

# Autoencoder training loop
print("Starting Autoencoder Training...")

ae_train_losses = []
ae_val_losses = []

for epoch in range(autoencoder_epochs):
    running_loss = 0.0
    ldm.train()
    # Training loop
    for batch_idx, (batch, _) in enumerate(train_loader):
        batch = batch.to(device)
        autoencoder_optimizer.zero_grad()
        recon = ldm.autoencoder_forward(batch)
        # print(batch.size(),recon.size())
        loss = reconstruction_loss(recon, batch)
        loss.backward()
        autoencoder_optimizer.step()
        running_loss += loss.item()

    # Validation loop
    ldm.eval()
    val_loss = 0.0
    num_val_batches = 0
    with torch.no_grad():
        for batch, _ in val_loader:
            batch = batch.to(device)
            recon = ldm.autoencoder_forward(batch)
            val_loss += reconstruction_loss(recon, batch).item()
            num_val_batches += 1

    avg_val_loss = val_loss / num_val_batches
    print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.6f}")

    print("-" * 50)
    ae_train_losses.append(running_loss / len(train_loader))
    ae_val_losses.append(avg_val_loss)
    autoencoder_scheduler.step(avg_val_loss)

store_path = "ldm_results" if not args.cold_diff else "ldm_results_cold_diff"
if not os.path.exists(store_path):
    os.makedirs(store_path, exist_ok=True)


print("Autoencoder Training Completed!")

# Save the losses
torch.save(
    {
        "ae_train_losses": ae_train_losses,
        "ae_val_losses": ae_val_losses,
    },
    f"{store_path}/autoencoder_losses_{steps}_{latent_channels}.pth",
)

print("\nStarting Diffusion Model Training...")


# Freeze autoencoder weights
ldm.freeze_autoencoder()

# Diffusion training loop
diff_train_losses = []
diff_val_losses = []

for epoch in range(diffusion_epochs):
    running_loss = 0.0
    ldm.train()
    # Training loop
    for batch_idx, (batch, _) in enumerate(train_loader):
        batch = batch.to(device)
        diffusion_optimizer.zero_grad()
        t = diffusion.sample_timesteps(batch.shape[0]).to(device)
        output = ldm(batch, t)
        loss = loss_fn(output, batch)
        loss.backward()
        diffusion_optimizer.step()
        running_loss += loss.item()

    # Validation loop
    ldm.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch, _ in val_loader:
            batch = batch.to(device)
            t = diffusion.sample_timesteps(batch.shape[0]).to(device)
            output = ldm(batch, t)
            val_loss += loss_fn(output, batch).item()

    avg_val_loss = val_loss / len(val_loader)
    diff_val_losses.append(avg_val_loss)
    # Plot images
    if epoch % 10 == 0:
        print(f"Epoch {epoch}")
        # plot_images(batch)
        # Sample images
        sampled_images_list = []
        num_samples = 1000
        batch_size_sampling = 64
        for i in range(0, num_samples, batch_size_sampling):
            curr_batch_size = min(batch_size_sampling, num_samples - i)
            rand_x = torch.randn((curr_batch_size, 1, 28, 28)).to(device)
            t = torch.ones(curr_batch_size, dtype=torch.long).to(device) * steps
            sample_images = ldm(rand_x, t)
            sampled_images = (sample_images.clamp(-1, 1) + 1) / 2  # Normalize to [0, 1]
            sampled_images_list.append(sampled_images.cpu())
        all_sampled_images = torch.cat(sampled_images_list, dim=0)
        # save_path = f"ldm_results/sampled_images/{steps}_{latent_channels}_epoch_{epoch}.png"
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save individual sampled images
        for idx, img in enumerate(all_sampled_images[:1000]):
            img_save_path = f"{store_path}/sampled_images/{steps}_{latent_channels}_{args.blur_sigma}_epoch_{epoch}/img_{idx}.png"
            os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
            save_image(img, img_save_path)
        print(
            f"Sampled images saved to {store_path}/sampled_images/{steps}_{latent_channels}_{args.blur_sigma}_epoch_{epoch}/"
        )
    print(
        f"Epoch {epoch} Avg val loss:{avg_val_loss}, lr = {diffusion_optimizer.param_groups[0]['lr']:.6f}"
    )
    diff_train_losses.append(running_loss / len(train_loader))
    print("-" * 50)
    diffusion_scheduler.step(avg_val_loss)


print("Diffusion Model Training Completed!")

os.makedirs(store_path, exist_ok=True)
# Save the losses
torch.save(
    {
        "diff_train_losses": diff_train_losses,
        "diff_val_losses": diff_val_losses,
    },
    f"{store_path}/ldm_diffusion_losses_{steps}_{args.blur_sigma}_{latent_channels}.pth",
)

# Define the directory for saving models
run_number = f"{steps}_{latent_channels}_{args.blur_sigma}"
save_dir = f"{store_path}/models/{run_number}/"
os.makedirs(save_dir, exist_ok=True)

# Save the model
torch.save(ldm.state_dict(), os.path.join(save_dir, "latent_diffusion_model.pth"))

# Save the model architecture
torch.save(ldm, os.path.join(save_dir, "latent_diffusion_model_full.pth"))

# Save the encoder and decoder separately
torch.save(encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
torch.save(decoder.state_dict(), os.path.join(save_dir, "decoder.pth"))

# Save the UNet model
torch.save(unet.state_dict(), os.path.join(save_dir, "unet.pth"))


#
