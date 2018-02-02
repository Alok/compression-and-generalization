#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy

import torch.nn.functional as F
from torch import nn

from cli_options import args
from data import NUM_CLASSES, input_dim


def flatten(t):
    return t.view(-1, input_dim)


class Net(nn.Module):
    def __init__(self, hidden_dim=512):
        H = self.hidden_dim = hidden_dim
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, H)
        self.fc2 = nn.Linear(H, H)
        self.out = nn.Linear(H, NUM_CLASSES)

    def forward(self, x):
        x = flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.out(x))
        return x


# Use same weights to ensure they start exactly the same
real_model = Net()

fake_model = Net()
fake_model.load_state_dict(real_model.state_dict())

if args.cuda:
    real_model = nn.DataParallel(real_model.cuda())
    fake_model = nn.DataParallel(fake_model.cuda())

if __name__ == '__main__':
    f = real_model
