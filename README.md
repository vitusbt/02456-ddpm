# Implementation of Denoising Diffusion Probabilistic Models
This repository contains the code for our project in the course 02456 Deep learning, where we have implemented both Denoising Diffusion Probabilistic Models (DDPM) and Classifier-Free Guidance (CFG) on the MNIST and CIFAR-10 datasets.

The important files are `ddpm.py` and `ddpm_cfg.py`. The code for both files is also collected in Jupyter notebook format in `ddpm_notebook.ipynb`. The code automatically downloads the datasets when run. Note that the code is intented to be run on GPU as otherwise it is extremely slow.

The folder `samples` contains all the samples generated from our experiments. Note that this folder is NOT necessary to run the code.
