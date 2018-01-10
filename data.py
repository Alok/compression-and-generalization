#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
import os
import random
import re
import shutil
import subprocess
import sys
from logging import debug, info, log
from pathlib import Path

import better_exceptions
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torchvision.transforms as transforms
from keras.datasets import mnist
from keras.utils import to_categorical
from pudb import set_trace
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets

from cli_options import args

# get data into nice,normalized numpy arrays

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 784).astype(np.float32)
x_test = x_test.reshape(-1, 784).astype(np.float32)

# normalize to range (0,1)
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


def create_loader(data: np.ndarray, labels: np.ndarray, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(data), torch.from_numpy(labels)),
        batch_size=args.batch_size,
        shuffle=shuffle,
        pin_memory=False,
    )


##############################################################################
# Returns: data loaders

real_train_loader = create_loader(x_train, y_train)
real_test_loader = create_loader(x_test, y_test)

# random (0,1) noise
fake_train_loader = create_loader(np.random.random_sample(x_train.shape), y_train)
fake_test_loader = create_loader(np.random.random_sample(x_test.shape), y_test)

# TODO rm the use of keras and just directly modify the pytorch mnist dataloader
# TODO try on cifar and imagenet
