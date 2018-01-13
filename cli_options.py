#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST')

parser.add_argument(
    '--batch-size',
    type=int,
    default=128,
    metavar='N',
    help='input batch size for training (default: 128)',
)

parser.add_argument(
    '--epochs',
    '-e',
    type=int,
    default=100,
    metavar='N',
    help='number of epochs to train (default: 100)',
)

parser.add_argument(
    '--lr',
    type=float,
    default=0.01,
    metavar='LR',
    help='learning rate (default: 0.01)',
)

parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    metavar='M',
    help='SGD momentum (default: 0.9)',
)

parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training',
)

parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)',
)

parser.add_argument(
    '--iters',
    '-i',
    type=int,
    default=10,
)


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if __name__ == '__main__':
    pass
