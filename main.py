#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import torch

import compress
from cli_options import args
from data import DataLoader, GenDataset, fake_loader, real_loader
from model import fake_model, real_model
from plot import plot
from train import train

# Don't plot interactively
plt.ioff()

ITERS = 10
REAL_EPOCHS = args.epochs
FAKE_EPOCHS = 1 * REAL_EPOCHS

# Set seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.cuda:
    kwargs = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0, 'pin_memory': False}
else:
    kwargs = {}

# The first model to be trained is the uncompressed version.
real = real_model
fake = fake_model

for i in range(ITERS):

    # generate training data for compressed models (except when training the real model)
    if i != 0:
        real_loader = DataLoader(GenDataset(real_model), **kwargs)
        fake_loader = DataLoader(GenDataset(fake_model), **kwargs)

    # Train
    real_losses = train(
        real,
        loader=real_loader,
        epochs=REAL_EPOCHS,
    )

    fake_losses = train(
        fake,
        loader=fake_loader,
        epochs=FAKE_EPOCHS,
    )

    # Plot performance
    plt = plot(
        iteration=i,
        real=real_losses,
        fake=fake_losses,
    )

    # save models
    torch.save(real, 'real-%s.pt' % i)
    torch.save(fake, 'fake-%s.pt' % i)

    # # Compress and repeat
    real = compress.datagen_compress(real)
    fake = compress.datagen_compress(fake)
