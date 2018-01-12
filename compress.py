#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader

from cli_options import args
from data import GenDataset
from model import Net
from train import train

# TODO try pruning instead of masking. Drop rows before and after layer you want to prune.


def mask_compress(f, THRESHOLD=0.05):

    # Faster than `deepcopy` and we only need the weights
    g = Net()
    g.load_state_dict(f.state_dict())

    g = torch.nn.DataParallel(g.cuda()) if args.cuda else g
    # Mask weights/biases
    for l in g.parameters():
        # 1 if larger than `THRESHOLD`, 0 otherwise
        mask = (torch.abs(l.data) > THRESHOLD).float()
        l.data.mul_(mask)

    # TODO make this work on individual connection level rather than whole layer
    # # ensure zero-weight parameters are never updated to mimic pruning
    # for param in g.parameters():
    #     if param == 0:
    #         param.requires_grad = False

    return g


def datagen_compress(f):
    # Init smaller network that uses 20% fewer nodes in the hidden layers

    hidden_dim = f.module.hidden_dim if args.cuda else f.hidden_dim
    g = Net(hidden_dim=int(hidden_dim * 0.8))

    g = torch.nn.DataParallel(g.cuda()) if args.cuda else g

    return g


if __name__ == '__main__':
    h = mask_compress(Net())
