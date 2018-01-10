#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import torch.nn as nn
from data import fake_test_loader, real_test_loader
from torch.autograd import Variable as V

from cli_options import args


def eval(f, real=True) -> List[float]:

    losses = []  # losses for each batch
    f.eval()  # put the model in evaluation (test) mode
    criterion = nn.NLLLoss()
    loader = real_test_loader if real else fake_test_loader

    for i, (x, y) in enumerate(loader):

        if args.cuda:
            x, y = x.cuda(), y.cuda()
        x, y = V(x), V(y)

        out = f(x)

        loss = criterion(out, y)

        losses.append(loss)
    return losses


if __name__ == '__main__':
    pass
