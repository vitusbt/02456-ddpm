import sys
import os

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

import numpy as np
import matplotlib.pyplot as plt

from torch.nn.parameter import Parameter
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torchvision.utils import make_grid

#!pip install labml-nn
from labml_nn.diffusion.ddpm.unet import UNet

import utils

## Hyperparameters
BATCH_SIZE = 128
T = 1000
UNET_N_CHANNELS = 64 # must be a multiple of 32
LR = 0.00002
LR_SCHEDULER_STEP_SIZE = 20
LR_SCHEDULER_GAMMA = 0.5
N_EPOCHS = 60
SAMPLE_BATCH_SIZE = 256
N_SAMPLE_BATCHES = 50
DATASET = 'MNIST'
run_name = 'mnist_final3'

# Print the hyperparameters
print(f'BATCH_SIZE={BATCH_SIZE}')
print(f'T={T}')
print(f'UNET_N_CHANNELS={UNET_N_CHANNELS}')
print(f'LR={LR}')
print(f'LR_SCHEDULER_STEP_SIZE={LR_SCHEDULER_STEP_SIZE}')
print(f'LR_SCHEDULER_GAMMA={LR_SCHEDULER_GAMMA}')
print(f'N_EPOCHS={N_EPOCHS}')
print(f'SAMPLE_BATCH_SIZE={SAMPLE_BATCH_SIZE}')
print(f'N_SAMPLE_BATCHES={N_SAMPLE_BATCHES}')
print(f'DATASET={DATASET}')
print(f'run_name={run_name}')

# Load dataset
if DATASET == 'MNIST':
    IMG_CHANNELS = 1
    IMG_SIZE=32
    dataset = utils.load_MNIST()
elif DATASET == 'CIFAR10':
    IMG_CHANNELS = 3
    IMG_SIZE=32
    dataset = utils.load_CIFAR10()
else:
    raise ValueError('Dataset must be MNIST or CIFAR10')

device = "cuda" if torch.cuda.is_available() else "cpu"

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

# Define beta schedule
betas = linear_beta_schedule(timesteps=T).to(device)                            # Variance of q(x_t|x_{t-1})

# Pre-calculate different terms for closed form
alphas = 1. - betas                                                             # 1 - beta
alphas_cumprod = torch.cumprod(alphas, axis=0)                                  # alpha_bar
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)             
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)                                    # 
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)                                # mean of q(x_t|x_0)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)                 # variance of q(x_t|x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # posterior variance q(x_{t-1}|x_t,x_0)

def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device=device):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

model = UNet(image_channels=IMG_CHANNELS, n_channels=UNET_N_CHANNELS)
model.to(device)

print("Num params: ", sum(p.numel() for p in model.parameters()))

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.mse_loss(noise, noise_pred)

@torch.no_grad()
def sample_timestep(x, t, noise=None):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t[0] == 0:
        return model_mean
    else:
        if noise is None:
            noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

model.to(device)
optimizer = Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.StepLR(optimizer, LR_SCHEDULER_STEP_SIZE, gamma=LR_SCHEDULER_GAMMA)

epoch_loss = np.zeros(N_EPOCHS)

# Training loop
for epoch in range(N_EPOCHS):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
        loss = get_loss(model, batch[0], t)
        loss.backward()
        optimizer.step()

        epoch_loss[epoch] += loss.item()

        if np.isnan(loss.item()):
            raise Exception('Loss is NaN')

        if step % 50 == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")

    scheduler.step()

    epoch_loss[epoch] /= (step+1)
    print(f"Epoch {epoch} | Final Loss: {epoch_loss[epoch]} ")


# Save the loss after each epoch
np.save('epoch_loss_'+run_name+'.npy', epoch_loss)

@torch.no_grad()
def sample_batch():
    """
    Sample a batch of SAMPLE_BATCH_SIZE images
    """
    img_size = IMG_SIZE
    img = torch.randn((SAMPLE_BATCH_SIZE, IMG_CHANNELS, img_size, img_size), device=device)

    for i in range(0,T)[::-1]:
        t = torch.full((SAMPLE_BATCH_SIZE,), i, device=device, dtype=torch.long)
        noise = torch.randn_like(img)
        img = sample_timestep(img, t, noise)

    # Clamp the image to the range [-1,1]
    img = torch.clamp(img, -1.0, 1.0)

    return img

@torch.no_grad()
def sample_batch_steps(step=100):
    """
    Sample a batch of SAMPLE_BATCH_SIZE and save the image every 100 steps
    for visualizing the sampling process
    """
    img_size = IMG_SIZE
    img = torch.randn((SAMPLE_BATCH_SIZE, IMG_CHANNELS, img_size, img_size), device=device)

    img_steps = [img]

    for i in range(0,T)[::-1]:
        t = torch.full((SAMPLE_BATCH_SIZE,), i, device=device, dtype=torch.long)
        noise = torch.randn_like(img)
        img = sample_timestep(img, t, noise)
        if i % step == 0:
            img_steps.append(img)

    img_steps = torch.stack(img_steps, dim=1)
    img_steps = torch.clamp(img_steps, -1.0, 1.0)

    return img_steps


# Sample a bunch of images
from torchvision.utils import save_image

if not os.path.exists(f'samples_{run_name}'):
    os.makedirs(f'samples_{run_name}/all_samples')
    os.makedirs(f'samples_{run_name}/steps')
    os.makedirs(f'samples_{run_name}/grids')

# sample all batches
for batch in range(N_SAMPLE_BATCHES):
    print(f'sampling batch {batch}')
    imgs = sample_batch()
    imgs = (imgs + 1) * 0.5 # Transform to the range [0,1]
    for i, img in enumerate(imgs):
        save_image(img, f'samples_{run_name}/all_samples/batch{batch:03d}_img{i:03d}.png')
    img_grid = make_grid(imgs, nrow=int(np.sqrt(SAMPLE_BATCH_SIZE)))
    save_image(img_grid, f'samples_{run_name}/grids/batch{batch:03d}_grid.png')

# Sample a single batch where we show the steps
print('sampling batch with steps')
imgs_steps = sample_batch_steps()
imgs_steps = (imgs_steps + 1) * 0.5 # Transform to the range [0,1]
for i, img in enumerate(imgs_steps):
    img_step_sequence = make_grid(img, 11)
    save_image(img_step_sequence, f'samples_{run_name}/steps/steps{i:03d}.png')



