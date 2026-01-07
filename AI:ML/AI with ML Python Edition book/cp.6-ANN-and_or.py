import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "perceptron_module", "cp.6-ANN-perceptron.py"
)
perceptron_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perceptron_module)
perceptron = perceptron_module.perceptron
plot_hyperplane2d = perceptron_module.plot_hyperplane2d

import matplotlib.pyplot as plt
import numpy as np

# And problem
X = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
T = [1, -1, -1, -1]
X = np.array(X)
X = np.hstack((X, np.ones((len(X), 1))))
w = perceptron(X, T)
plot_hyperplane2d(X, T, w)
plt.show()
