#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

import torch

# TODO try pruning instead of masking. Drop rows before and after layer you want to prune.


def compress(f, THRESHOLD=0.05):

    # TODO return copy of model rather than modifying in-place
    # f = copy.deepcopy(f)

    # mask weights and biases
    for l in f.parameters():
        # 1 if larger, 0 otherwise
        mask = (torch.abs(l.data) > THRESHOLD).float()
        l.data.mul_(mask)

    # return f


if __name__ == '__main__':
    pass
