#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def plot(real=None, fake=None):

    # green for real, red for fake
    if real is not None:
        plt.plot(np.arange(len(fake)), real, color='g')
    if fake is not None:
        plt.plot(np.arange(len(fake)), fake, color='r')

    return plt


if __name__ == '__main__':
    plt.ioff()
