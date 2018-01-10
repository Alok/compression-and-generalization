#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

import torch


# TODO return copy of model
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
