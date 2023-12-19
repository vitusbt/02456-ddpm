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
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

# Functions for loading the datasets

def load_MNIST(transform=True):
    """Load the MNIST dataset"""
    if transform:
        data_transforms = [
            transforms.ToTensor(), # Scales data into [0,1]
            transforms.Pad(2, fill=0, padding_mode='constant'),
            transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
        ]
        data_transform = transforms.Compose(data_transforms)
    else:
        data_transforms = [
            transforms.ToTensor(), # Scales data into [0,1]
            transforms.Pad(2, fill=0, padding_mode='constant'),
        ]
        data_transform = transforms.Compose(data_transforms)

    trainset = MNIST("./temp/", train=True, download=True, transform=data_transform)
    testset = MNIST("./temp/", train=False, download=True, transform=data_transform)

    return torch.utils.data.ConcatDataset([trainset, testset])

def load_CIFAR10(transform=True):
    """Load the CIFAR10 dataset"""

    if transform:
        data_transforms = [
            transforms.ToTensor(), # Scales data into [0,1]
            transforms.Lambda(lambda t: (t * 2) - 1), # Scale between [-1, 1]
            transforms.RandomHorizontalFlip(p=0.5)
        ]
        data_transform = transforms.Compose(data_transforms)
    else:
        data_transform = transforms.ToTensor()

    trainset = CIFAR10("./temp_cifar10/", train=True, download=True, transform=data_transform)
    testset = CIFAR10("./temp_cifar10/", train=False, download=True, transform=data_transform)

    return torch.utils.data.ConcatDataset([trainset, testset])