#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import randint

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from cli_options import args

NUM_CLASSES = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1)),  # want (0,1) values
])

real_dataset = MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform,
)

fake_dataset = MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform,
)

fake_dataset.train_labels.random_(0, to=NUM_CLASSES)  # randomize labels

# TODO try on CIFAR and ImageNet
real_train_loader = DataLoader(
    real_dataset,
    batch_size=args.batch_size,
    shuffle=True,
)

fake_train_loader = DataLoader(
    fake_dataset,
    batch_size=args.batch_size,
    shuffle=True,
)

if __name__ == '__main__':
    pass
