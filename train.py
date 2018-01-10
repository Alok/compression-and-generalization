#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as V

from cli_options import args
from data import fake_train_loader, real_train_loader


def train(f, epochs=3, real=True):

    losses = []

    f.train()  # put the model in training mode
    criterion = nn.NLLLoss()
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

            losses.append(float(loss.data[0]))  # extract number for singleton tensor
    return losses
