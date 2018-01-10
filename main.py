#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import matplotlib.pyplot as plt
import torch

from cli_options import args
from compress import compress
from model import fake_model, real_model
from plot import plot
from train import train

ITERS = 5
# Set seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Don't plot interactively
plt.ioff()

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

for i in range(ITERS):

    # train and plot performance
    real_losses = train(real_model, real=True)
    fake_losses = train(fake_model, real=False)
    plt = plot(real_losses, fake_losses)
    plt.savefig(f'{i}.png')
    plt.gcf().clear()

    # compress and repeat
    compress(real_model)
    compress(fake_model)
