#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


# TODO return copy of model
def compress(f, THRESHOLD=0.05):
    # mask weights and biases
    for l in f.parameters():
        # 1 if larger, 0 otherwise
        mask = (torch.abs(l.data) > THRESHOLD).float()
        l.data.mul_(mask)


if __name__ == '__main__':
    pass
