# Kernal
import time

import numpy as np


class Kernel:
    def __init__(self, X, Y=None, kernel="rbf", degree=3, gamma=1, a=1, b=1):
        self.X = X
        self.degree = degree
        self.gamma = gamma
        self.a = a
        self.b = b
        if Y is None:
            self.Y = X
        else:
            self.Y = Y
        self.K = eval("self." + kernel.lower() + "()")

    def linear(self):
        return self.X @ self.Y.T

    def poly(self):
        return (self.linear() + 1) ** self.degree

    def rbf(self):
        XX = np.sum(self.X**2, axis=1)[:, None] @ np.ones((1, len(self.Y)))
        YY = np.sum(self.Y**2, axis=1)[:, None] @ np.ones((1, len(self.X)))
        K = XX + YY.T - 2 * self.linear()
        return np.exp(-K / self.gamma)

    def sigmoid(self):
        return np.tanh(self.a * self.linear() + self.b)
