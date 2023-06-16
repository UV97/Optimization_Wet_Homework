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


def calc_x_matrix(x, order=3):
    x_mat = []
    x_mat.append(np.ones([len(x)]))
    for i in range(1, order+1):
        x_mat.append(x**i)
    return np.stack(x_mat, axis=-1)


def calc_gradient(X, y, a):
    return (1/len(y)) * X.transpose() @ (X @ a - y)


def max_eig(matrix):
    return np.max(np.linalg.eigvals(matrix))


def min_eig(matrix):
    return np.min(np.linalg.eigvals(matrix))


def calc_rmse(h, h_star):
    return np.sqrt(np.sum((h - h_star)**2) / len(h))


def estimation_solve(x, y):
    X = calc_x_matrix(x)
    return np.linalg.inv(X.transpose() @ X) @ X.transpose() @ y
