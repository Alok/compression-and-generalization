#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from numpy import prod
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

from cli_options import args

NUM_CLASSES = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1)),  # want (0,1) values
])

real_dataset = MNIST(
    root='data/',
    train=True,
    download=True,
    transform=transform,
)

num_samples = int(real_dataset.train_data.size()[0])
input_dim = int(prod(real_dataset.train_data.size()[1:]))

fake_dataset = MNIST(
    root='data/',
    train=True,
    download=True,
    transform=transform,
)

fake_dataset.train_labels.random_(0, to=NUM_CLASSES)  # randomize labels

# TODO try on CIFAR and ImageNet
real_loader = DataLoader(
    real_dataset,
    batch_size=args.batch_size,
    shuffle=True,
)

fake_loader = DataLoader(
    fake_dataset,
    batch_size=args.batch_size,
    shuffle=True,
)


class GenDataset(Dataset):
    def __init__(self, f):
        # Use (0,1) uniform data since images are normalized to be in that range anyway.
        self.data_tensor = torch.Tensor(num_samples, input_dim).uniform_(0, 1)
        self.target_tensor = f(Variable(self.data_tensor)).data.max(1)[1]  # argmax

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return len(self.data_tensor)


if __name__ == '__main__':
    pass
