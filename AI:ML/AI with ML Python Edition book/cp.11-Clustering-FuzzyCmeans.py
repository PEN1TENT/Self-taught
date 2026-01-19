# Fuzzy C-means
import numpy as np
from matplotlib import pyplot as plt 
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("Kmeans", "cp.11-Clustering-Kmeans.py")
Kmeans_module = importlib.util.module_from_spec(spec)
sys.modules["Kmeans_module"] = Kmeans_module
spec.loader.exec_module(Kmeans_module)
plot = Kmeans_module.plot

def FuzzyCmeans(X, K, m = 2, th = 1e-6, show = False):
    # initial centroids by randomly selecting from dataset 
    centroids = X[np.random.choice(len(X), 3, replace = False)]
    if X.shape[1] > 3:
        show = False # can plot onlt 1D, 2D and 3D spaces
    i = 0
    while True:
        i += 1
        # Distances
        D = np.zeros((K, len(X)))
        for k in range(K):
            D[k] = np.sum((X - centroids[k]) ** 2, axis = 1)
            # Memberships 
            u = D ** (1 / (m - 1))
            u = 1 / u # zero distances make Inf 
            u /= np.sum(u, axis = 0) # Inf makes NaN
            u[np.isnan(u)] = 1 # Replace NaN with 1 
            # Update centroids 
            old_centroids = centroids.copy()
        for k in range(K):
            temp = u[k] ** m
            centroids[k] = np.sum(X.T * temp.T, axis = 1) / np.sum(temp)
            # use transpose for broadcast trick 
        if show:
            print(i)
            print(centroids)
            if X.shape[1] <= 3:
                plt.clf()
                plot(X, centroids)
                plt.pause(.5)
        if np.sum(np.abs(old_centroids - centroids)) <= th:
            if show:
                print('done')
            if show and X.shape[1] <= 3:
                plt.show()
            return np.argmin(D, axis = 0), centroids

if __name__ == '__main__':
    K = 3
    N = 500 
    d = 2 
    X = np.empty((0, d))
    s = [0.1, 0.3, 0.15]
    m = [[2, .2, 1], [.2, 1.4, 2], [1.1, .6, 1]]
    for k in range(K):
        X = np.vstack((X, s[k] * np.random.randn(N // K, d) + m[k][:d]))
        
    cluster, centroids = FuzzyCmeans(X, K, show = True)