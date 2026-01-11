import importlib.util
import sys

spec = importlib.util.spec_from_file_location("SVM_SMO_module", "cp.7-SVM-SVM_SMO.py")
SVM_SMO = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SVM_SMO)
SVM = SVM_SMO.SVM

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

# Generate data
dim = 2  # Dimension
N = 50  # Number of samples
X = np.vstack(
    (np.random.rand(N // 2, dim) - 0.4, np.random.rand(N // 2, dim) + 0.4)
)  # data
y = np.hstack((np.ones((N // 2)), -np.ones((N // 2))))  # labels

# SVM
svm = SVM(C=100, kernel="rbf")
svm.train(X, y)

# Plot data
plt.plot(X[: N // 2, 0], X[: N // 2, 1], "g*")
plt.plot(X[N // 2 :, 0], X[N // 2:, 1], "ro")
# Plot Suport Vectors
plt.plot(X[svm.iSV, 0], X[svm.iSV, 1], "ks", mfc="none")

# Plot hyperplane
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
x1, x2 = np.meshgrid(
    np.arange(xlim[0], xlim[1], 0.01), np.arange(ylim[0], ylim[1], 0.01)
)
Z = np.vstack((x1.flatten(), x2.flatten())).T
fx = svm.predict(Z)
fx = fx.reshape(x1.shape)
plt.contour(x1, x2, fx, 50, cmap=cm.coolwarm)
cs = plt.contour(x1, x2, fx, levels=[-1, 0, 1], colors=["k", "k", "k"])
plt.clabel(cs, fmt="%d", colors="k", fontsize=14)

# Plot alpha
allalpha = np.zeros(len(X))
allalpha[svm.iSV] = svm.alpha
plt.figure()
plt.plot(allalpha, ".")
plt.xlabel("index")
plt.ylabel(r"$\alpha$")
plt.show()
