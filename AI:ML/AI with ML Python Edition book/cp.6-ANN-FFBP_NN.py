# Feedforward Backpropagation Neural Network
import os
import sys

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "FFBP_sigmoid_module", "cp.6-ANN-FFBP_sigmoid.py"
)
FFBP_sigmoid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FFBP_sigmoid)
FFBP = FFBP_sigmoid.FFBP

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
T = np.array([0, 1, 1, 0])
hidden = [2]

net = FFBP(X, T, hidden, lr=0.1, alpha=0.8, eps=0.001, epochs=5000)

plt.plot(net.error)
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.show()

# Plot hyperplane
x = np.arange(0, 1.01, 0.01)
y = np.arange(0, 1.01, 0.01)
output = np.zeros((len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        output[i, j] = net.forward(np.array([[x[i], y[j]]]))[0][0]
x, y = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(x, y, output, cmap=cm.cool, linewidth=0, antialiased=False)
ax.scatter(X[:, 0][T == 0], X[:, 1][T == 0], T[T == 0], color="g", s=50)
ax.scatter(X[:, 0][T == 1], X[:, 1][T == 1], T[T == 1], color="r", s=50)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Output")
plt.show()
