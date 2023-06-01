import numpy as np
from matplotlib import pyplot as plt


def calculate_poly(x, a):
    y = np.zeros(x.shape)
    for i in range(len(a)):
        y += a[i] * (x**i)
    return y


def plot_poly(a, start, end, graph_label):
    x = np.linspace(start, end)
    y = calculate_poly(x, a)
    plt.plot(x, y, label=graph_label)
