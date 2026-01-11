import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers
from matplotlib import cm
from pandas.core.dtypes.dtypes import BDay

spec = importlib.util.spec_from_file_location("Kernel_module", "cp.7-SVM-Kernel.py")
Kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Kernel)
Kernel = Kernel.Kernel

# Generate data
dim = 2  # Dimension
N = 50  # Number of samples
X = np.vstack(
    (np.random.rand(N // 2, dim) - 0.4, np.random.rand(N // 2, dim) + 0.4)
)  # data
y = np.hstack((np.ones((N // 2)), -np.ones((N // 2))))[None, :]  # labels

# Quadratic Programming
kernel = "rbf"
C = 10
Q = Kernel(X, kernel=kernel).K * (y.T @ y)
c = -np.ones(N)

l = np.zeros(N)
u = np.ones(N)
A = np.vstack((np.eye(N), -np.eye(N)))
b = np.concatenate((u, -l))

E = y
d = 0.0

alpha = solvers.qp(matrix(Q), matrix(c), matrix(A), matrix(b), matrix(E), matrix(d))
alpha = np.array(alpha["x"]).flatten()

# Find b
sv = alpha > 0.5
x_1 = X[(y == 1).flatten() & sv]
b = 1 - (y * alpha) @ Kernel(X, Y=x_1, kernel=kernel).K
b = np.min(b.flatten())

# Plot data
plt.plot(X[: N // 2, 0], X[: N // 2, 1], "g*")
plt.plot(X[N // 2 :, 0], X[N // 2 :, 1], "ro")

# Plot hyperplane
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
x1, x2 = np.meshgrid(
    np.arange(xlim[0], xlim[1], 0.01), np.arange(ylim[0], ylim[1], 0.01)
)
Z = np.vstack((x1.flatten(), x2.flatten())).T
fx = (y * alpha) @ Kernel(X, Z, kernel=kernel).K + b
fx = fx.reshape(x1.shape)
plt.contour(x1, x2, fx, 50, cmap=cm.coolwarm)
cs = plt.contour(x1, x2, fx, levels=[-1, 0, 1], colors=["k", "k", "k"])
plt.clabel(cs, fmt="%d", colors="k", fontsize=14)
# Plot Support Vectors
plt.plot(X[sv, 0], X[sv, 1], "ks", mfc="none")

# Plot alpha
plt.figure()
plt.plot(alpha, ".")
plt.xlabel("index")
plt.ylabel(r"$\alpha$")
plt.show()
