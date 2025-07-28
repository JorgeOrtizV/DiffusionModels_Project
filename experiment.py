import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision.transforms import GaussianBlur
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter, zoom, mean as ndimage_mean

import os
from pathlib import Path
import torchvision

import joblib
from sklearn.mixture import GaussianMixture
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import argparse


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


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
        if self.blur_sigma <= 0:
            return x0
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
        # print(x.size())
        x = self.up(x)
        # print(x.size())
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


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(
        torch.cat([torch.cat([i for i in images.detach().cpu()], dim=-1)], dim=-2)
        .permute(1, 2, 0)
        .cpu(),
        cmap="gray",
    )
    plt.show()


transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((28, 28)),  # Resize to 28x28 for MNIST
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)


def split_indices(n, val_pct):
    n_val = int(val_pct * n)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]


# TRAINING SETUP

np.random.seed(32)

EPOCHS = 100
batch_size = 32
image_size = 28
learning_rate = 1e-3
loss_type = "L1"  # 'L1', 'L2', 'SmoothL1', 'Huber'
sampling_type = "gmm"  # 'random', 'black', 'real_degraded', 'gmm'

LIMIT = True
num_samples = 6016  # FOR LIMIT
FASHION_MNIST = False

# DEGRADATION TYPE
degradation_type = "blur"  # 'blur', 'pixellate', 'inpainting', 'snow', 'gaussian'

# Steps for diffusion process - changes based on the function used
# For pixellate, steps = 4 (MNIST) or 6 (CIFAR10)
# Parse arguments for steps and blur_sigma
parser = argparse.ArgumentParser(description="Set diffusion parameters.")
parser.add_argument("--steps", type=int, default=200, help="Number of diffusion steps.")
parser.add_argument(
    "--blur_sigma", type=float, default=0.33, help="Sigma value for Gaussian blur."
)
args = parser.parse_args()

steps = args.steps
blur_sigma = args.blur_sigma

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
if LIMIT:

    num_test_samples = num_samples // 6

    # Randomly select indices for train and test
    all_indices = np.random.permutation(len(dataset))
    train_indices = all_indices[:num_samples]
    test_indices = np.random.permutation(len(test_dataset))[:num_test_samples]

    dataset = torch.utils.data.Subset(dataset, train_indices)
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

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=5
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

diffusion = coldDiff(
    size=image_size,
    steps=steps,
    degradation_type=degradation_type,
    blur_sigma=blur_sigma,
)
noise_function = diffusion.degradation
length = len(train_loader)
print(length)

train_losses = []
val_losses = []

# FIT A GMM

FIT_GMM = True

if FIT_GMM:
    all_blurred = []

    for x, _ in train_loader:
        x = x.to(device)
        batch_size_t = x.size(0)

        t = (
            torch.full((batch_size_t,), steps).long().to(device)
        )  # Full degradation (T steps)
        x_blurred = diffusion.degradation(x, t)

        all_blurred.append(x_blurred.cpu())

    # Stack all blurred images
    all_blurred = torch.cat(all_blurred, dim=0)  # shape: [N, 1, 28, 28]
    all_blurred = all_blurred.squeeze(1)  # [N, 28, 28]

    X = all_blurred.reshape(all_blurred.shape[0], -1)  # [N, 784]

    # Choose number of components (1–5 is usually enough for MNIST)
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0)
    gmm.fit(X.numpy())  # convert to numpy

    # Save the GMM model
    print("Saving GMM model")
    Path("./gmm_models").mkdir(parents=True, exist_ok=True)
    joblib.dump(
        gmm,
        f"./gmm_models/gmm_mnist_{degradation_type}_{blur_sigma}_{steps}_{num_samples}.pkl",
    )

run_name = f"coldDiffusion_{degradation_type}_s{steps}_b{batch_size}_{sampling_type}_{learning_rate:.1e}_e{EPOCHS}_{loss_type}_{len(dataset)}_{'LIMITED' if LIMIT else 'ALLDATA'}_blur{blur_sigma}"
if scheduler is not None:
    run_name += f"_scheduler_{scheduler.__class__.__name__}"

# Create runs directory if it doesn't exist
if not os.path.exists("runs"):
    os.makedirs("runs")

# Add a version number to the run name if it already exists
version = 1
while os.path.exists("runs/" + run_name):
    run_name = f"coldDiffusion_{degradation_type}_s{steps}_b{batch_size}_{sampling_type}_{learning_rate:.1e}_e{EPOCHS}_{loss_type}_{len(dataset)}_{'LIMITED' if LIMIT else 'ALLDATA'}_blur{blur_sigma}"
    if scheduler is not None:
        run_name += f"_scheduler_{scheduler.__class__.__name__}"
    run_name += f"_v{version}"
    version += 1


writer = SummaryWriter("runs/" + run_name)
print(run_name)

# TRAINING LOOP
for epoch in range(EPOCHS):
    print("Epoch ", epoch)
    train_loss = 0
    for x, _ in tqdm(train_loader):
        optimizer.zero_grad()
        x = x.to(device)
        t = diffusion.sample_timesteps(x.shape[0]).to(device)
        x_t = noise_function(x, t)
        pred = model(x_t, t)
        loss = loss_fn(pred, x)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for x, _ in val_loader:

            # print(type(x))
            # print(x.shape)
            x = x.to(device)
            t = diffusion.sample_timesteps(x.shape[0]).to(device)
            x_t = noise_function(x, t)
            pred = model(x_t, t)
            batch_val_loss = loss_fn(pred, x)
            val_loss += batch_val_loss.item()

            del x, x_t, t, pred, batch_val_loss
        scheduler.step(val_loss / len(val_loader))
        # Sample only every n epochs for opt purposes
        if (epoch + 1) % 30 == 0:
            # Sample 1000 images and save them to "sample_images/run_name"
            num_samples = 1000
            samples_per_batch = 100
            output_dir = f"sample_images/{run_name}_EPOCH{epoch}"
            os.makedirs(output_dir, exist_ok=True)
            all_samples = []
            # print(f"Sampling {num_samples} images...")
            for i in range(0, num_samples, samples_per_batch):
                # print(i)
                sampled_images = diffusion.sample(
                    model,
                    batch_size=samples_per_batch,
                    initial_image=sampling_type,
                    gmm=gmm,
                    data_loader=val_loader,
                )
                for j, img in enumerate(sampled_images):
                    save_path = os.path.join(output_dir, f"sample_{i + j:04d}.png")
                    save_image(img, save_path)

            print(f"Saved {num_samples} samples to {output_dir}")

    # Save loss
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    writer.add_scalar("Loss/val", avg_val_loss, epoch)
    writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f} | Learning Rate: {scheduler.get_last_lr()[0]:.10f}"
    )
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)


writer.close()

# Save model checkpoint
checkpoint_path = f"checkpoints/{run_name}.pth"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
torch.save(model.state_dict(), checkpoint_path)
