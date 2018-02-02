#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


def plot(iteration, real=None, fake=None):

    # green for real, red for fake
    if real is not None:
        plt.plot(range(len(real)), real, color='g')
    if fake is not None:
        plt.plot(range(len(fake)), fake, color='r')

    plt.title(f'Iteration: {iteration}')
    plt.ylabel('Loss')
    plt.xlabel('Minibatch')

    plt.savefig((f'figs/{iteration}.png'))
    plt.gcf().clear()

    return plt


if __name__ == '__main__':
    plt.ioff()
