#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import torch

from cli_options import args
from compress import compress
from data import DataLoader, GenDataset, fake_loader, real_loader
from model import fake_model, real_model
from plot import plot
from train import train

ITERS = 5
REAL_EPOCHS = args.epochs
FAKE_EPOCHS = 10 * REAL_EPOCHS

STEP_SIZE = 50

# Set seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Don't plot interactively
plt.ioff()

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
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
        real=real_losses,
        fake=fake_losses,
    )
    plt.title('Iteration: %s' % iter)
    plt.ylabel('Loss')
    plt.xlabel('Minibatch')
    plt.savefig(('figs/%s.png' % iter))
    plt.gcf().clear()

    # # Compress and repeat
    compress(real_model)
    compress(fake_model)
