#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from cli_options import args


def train(f, loader, epochs=3):

    losses = []

    f.train()  # put the model in training mode
    criterion = nn.NLLLoss()

    optimizer = optim.SGD(f.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(epochs):
        for x, y in loader:

            if args.cuda:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)

            out = f(x)

            optimizer.zero_grad()
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.data[0]))  # extract number for singleton tensor
    return losses
