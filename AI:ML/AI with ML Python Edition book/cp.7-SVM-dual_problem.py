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

# Quadratic Programming
Q = (D @ X) @ (X.T @ D)
c = -np.ones(N)
lb = np.zeros(N)
A = -np.eye(N)
b = -lb
E = np.diag(D).reshape((1, len(D)))
d = 0.0
alpha = solvers.qp(matrix(Q), matrix(c), matrix(A), matrix(b), matrix(E), matrix(d))
alpha = np.array(alpha["x"]).flatten()

# Find w, b
w = np.sum((np.diag(D) * alpha)[:, None] * X, axis=0)
b = (np.min(X[: N // 2] @ w) + np.max(X[N // 2 :] @ w)) / -2

# Plot data
plt.plot(X[: N // 2, 0], X[: N // 2, 1], "g*")
plt.plot(X[N // 2 :, 0], X[N // 2 :, 1], "ro")

# Plot hyperplane
x = np.arange(np.min(X[:, 0]), np.max(X[:, 0]) + 0.01, 0.01)
fx = (-b - w[0] * x) / w[1]
plt.plot(x, fx, "b")

plt.figure()
plt.plot(alpha, ".")
plt.show()

# Plot Support Vectors
plt.figure()
plt.plot(X[alpha > 0.5, 0], X[alpha > 0.5, 1], "ks", mfc="none")
plt.show()
