#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy

import torch.nn.functional as F
from torch import nn

from cli_options import args


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28**2, 512)
        self.fc2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.out(x))
        return x


# Use deepcopy to ensure they start exactly the same
real_model = Net()
fake_model = copy.deepcopy(real_model)

if args.cuda:
    real_model = nn.DataParallel(real_model.cuda())
    fake_model = nn.DataParallel(fake_model.cuda())

if __name__ == '__main__':
    pass
