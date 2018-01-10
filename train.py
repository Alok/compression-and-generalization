#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import torch.nn as nn
import torch.optim as optim
from data import fake_train_loader, real_train_loader
from torch.autograd import Variable as V

from cli_options import args


def train(f, *, epochs=3, real=True) -> List[float]:

    losses = []

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

            losses.append(loss)
    return losses
