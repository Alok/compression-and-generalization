#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import random
import re
import shutil
import subprocess
import sys
from logging import debug, info, log
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from pudb import set_trace
from torch.autograd import Variable as V
from torch.nn import Parameter as P
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO layers
        self.fc1 = nn.Linear(28**2, 512)
        self.fc2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, 10)

    def forward(self, x):
        # TODO connect layers
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.softmax(self.out(x))
        return x


# TODO train with SGD with momentum param 0.9

net = Net()

THRESHOLD = 0.05


def compress(f: Net):

    # mask weights and biases
    for l in f.parameters():
        mask = (torch.abs(l.data) > THRESHOLD).float()
        l.data.mul_(mask)



f = compress(net)
