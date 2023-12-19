import sys
import os

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

# %matplotlib inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from torch.nn.parameter import Parameter
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from torchvision.utils import save_image

import utils

dataset = utils.load_MNIST(transform=False)

for i, (img, label) in enumerate(dataset):
    save_image(img, f'MNIST_png/{i:05d}.png')

# dataset = utils.load_CIFAR10(transform=False)

# for i, (img, label) in enumerate(dataset):
#     save_image(img, f'CIFAR10_png/{i:05d}.png')