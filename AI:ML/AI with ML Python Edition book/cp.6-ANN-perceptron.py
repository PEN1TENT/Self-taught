# Perceptron
import numpy as np
from matplotlib import pyplot as plt

def perceptron(X, T, lr = 0.1, w = None):
    if w is None:
        w = X[np.random.randint(0, len(X))].copy()
    X = np.array(X)
    done = False
    while not done:
        done = True 
        for i in range(len(X)):
            if T[i]*np.dot(X[i], w) <= 0:
                w += lr * T[i] * X[i]
                done = False
    return w

def plot_hyperplane2d(X, T, w):
    X = np.array(X)
    T = np.array(T)
    plt.plot(X[T == 1, 0], X[T == 1, 1], 'og')
    plt.plot(X[T == -1, 0], X[T == -1, 1], 'or')
    if w[1] != 0:
        xlim = plt.gca().get_xlim()
        slope = -w[0] / w[1]
        bias = 0
        if len(w) > 2:
            bias = -w[2] / w[1]
        plt.plot(xlim, [xlim[0] * slope + bias, xlim[1] * slope + bias], 'b')
    else:
        ylim = plt.gca().get_ylim()
        plt.plot([0, 0], ylim, 'b')
    
if __name__ == '__main__':
    # test problem
    X = [[1, 2], [2.5, 3], [3, 1], [-3, -1.7], [-1.6, -3], [1.5, -3]]
    T = [1, 1, 1, -1, -1, -1]
    
    # Initial by random
    w = perceptron(X, T)
    print(w)
    plot_hyperplane2d(X, T, w)
    plt.show()
    
    # All initial cases
    for i, x in enumerate(X):
        w = perceptron(X, T, w = x)
        plt.subplot(1, len(X), i + 1)
        plot_hyperplane2d(X, T, w)
    plt.show()