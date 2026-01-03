# K-d tree Performance
import importlib

import numpy as np

kdtree_module = importlib.import_module("cp_4_kNN_kdtree")
kdtree = kdtree_module.kdtree
import importlib.util
import sys
import time

spec = importlib.util.spec_from_file_location(
    "kNN_kdtree_module", "cp.4-kNN-kNN_kdtree.py"
)
kNN_kdtree_module = importlib.util.module_from_spec(spec)
sys.modules["kNN_kdtree_module"] = kNN_kdtree_module
spec.loader.exec_module(kNN_kdtree_module)
kNN_kdtree = kNN_kdtree_module.kNN_kdtree
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

MAX = 100
t = np.zeros((MAX, MAX))
for D in range(1, MAX + 1):
    print("dimension = ", D)
    for N in range(1, MAX + 1):
        X = np.random.rand(N, D)
        Y = np.arange(1, N + 1)
        t1 = time.time()
        repeat = 100
        for i in range(repeat):
            node = kdtree(X, Y)
            Ztest = kNN_kdtree(node, X[0, :].reshape((1, D)), 1)
        t[D - 1, N - 1] = (time.time() - t1) / repeat

# 3D plot
x = np.arange(len(t))
y = np.arange(len(t))
x, y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
surf = ax.plot_surface(x, y, t, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel("Dimension")
ax.set_ylabel("#Sample")
ax.set_zlabel("Computational time (s)")
plt.show()
