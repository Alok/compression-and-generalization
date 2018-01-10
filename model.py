#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO layers
        self.fc1 = nn.Linear(28**2, 512)
        self.fc2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, 10)

    def forward(self, x):
        # TODO connect layers
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.out(x))
        return x


if __name__ == '__main__':
    pass