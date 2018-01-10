#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import random
import re
import shutil
import subprocess
import sys
from logging import debug, info, log
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from data import fake_train_loader, real_train_loader
from pudb import set_trace
from torch.autograd import Variable as V
from torch.nn import Parameter as P
from torch.utils.data import DataLoader

from cli_options import args


def train(f, epochs=5, real=True) -> None:

    f.train()  # put the model in training mode
    criterion = nn.CrossEntropyLoss()
    loader = real_train_loader if real else fake_train_loader

    optimizer = optim.SGD(f.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(epochs):
        for i, (x, y) in enumerate(loader):

            if args.cuda:
                x, y = x.cuda(), y.cuda()
            x, y = V(x), V(y)

            out = f(x)

            optimizer.zero_grad()
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
