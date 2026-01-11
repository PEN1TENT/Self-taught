import numpy as np
from cvxopt import matrix, solvers
from matplotlib import pyplot as plt

# Generate data
dim = 2  # Dimension
N = 50  # Number of samples
X = np.vstack(
    (np.random.rand(N // 2, dim) - 0.5, np.random.rand(N // 2, dim) + 0.5)
)  # data
D = np.diag(np.hstack((np.ones((N // 2)), -np.ones((N // 2)))))  # labels

# Homogeneous coordinates
X = np.hstack((X, np.ones((len(X), 1))))

# Quadratic Programming
Q = np.eye(dim + 1)
c = np.zeros(dim + 1)
A = -D @ X
b = -np.ones(N)
w = solvers.qp(matrix(Q), matrix(c), matrix(A), matrix(b))
w = np.array(w["x"]).flatten()

# Plot data
plt.plot(X[: N // 2, 0], X[: N // 2, 1], "g*")
plt.plot(X[N // 2 :, 0], X[N // 2 :, 1], "ro")

# Plot hyperplane
x = np.arange(np.min(X[:, 0]), np.max(X[:, 0]) + 0.01, 0.01)
fx = (-w[2] - w[0] * x) / w[1]
plt.plot(x, fx, "b")
plt.show()
