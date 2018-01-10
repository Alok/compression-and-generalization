#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def plot(real, fake):

    xs = np.arange(len(real))

    # green for real, red for fake
    # plt.scatter(xs, real, color='g')
    # plt.scatter(xs, fake, color='r')
    plt.plot(xs,real,color='g')
    plt.plot(xs,fake,color='r')

    return plt


if __name__ == '__main__':
    plt.ioff()
