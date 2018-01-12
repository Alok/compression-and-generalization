#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

import torch

# TODO try pruning instead of masking. Drop rows before and after layer you want to prune.


def mask_compress(f, THRESHOLD=0.05):

    # Faster than `deepcopy` and we only need the weights
    g = Net()
    g.load_state_dict(f.state_dict())

    g = torch.nn.DataParallel(g.cuda()) if args.cuda else g
    # Mask weights/biases
    for l in g.parameters():
        # 1 if larger than `THRESHOLD`, 0 otherwise
        mask = (torch.abs(l.data) > THRESHOLD).float()
        l.data.mul_(mask)

    # return f


if __name__ == '__main__':
    h = mask_compress(Net())
