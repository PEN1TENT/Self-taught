# Expectation Maximization (EM) algorithm 
# for Clustering with Gaussian Mixture Model (GMM)
import numpy as np
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("Kmeans", "cp.11-Clustering-Kmeans.py")
Kmeans_module = importlib.util.module_from_spec(spec)
sys.modules["Kmeans_module"] = Kmeans_module
spec.loader.exec_module(Kmeans_module)
plot = Kmeans_module.plot
Kmeans = Kmeans_module.Kmeans 

spec2 = importlib.util.spec_from_file_location("GMM", "cp.11-Clustering-GMM.py")
GMM_module = importlib.util.module_from_spec(spec2)
sys.modules["GMM_module"] = GMM_module
spec2.loader.exec_module(GMM_module)
gmm = GMM_module.gmm
gmmplot = GMM_module.gmmplot
gmmplot3 = GMM_module.gmmplot3

def GMM_EM(X, K, useKmeans = False, show = False):
    # Initialization
    Gmu = np.mean(X, axis = 0)
    n, d = X.shape
    Ez = np.zeros((K, n))
    mu = np.zeros((K, d))
    t = np.ones(K) / K
    covar = []
    for j in range(K):
        mu[j] = Gmu + np.random.randn(d)
        covar.append(np.eye(d))
    # K-means:
    if useKmeans:
        idx, mu = Kmeans(X, K)
        for j in range(K):
            covar[j] = np.cov(X[idx == j].T)
    if show:
        plot(X, mu)
    muold = mu.copy()
    T = 1e-6 
    diff = T + 1 
    while diff > T:
        # E-step 
        for j in range(K):
            Ez[j] = t[j] * gmm(X, mu[j], covar[j])
        Ez = Ez @ np.diag(1 / np.sum(Ez, axis = 0))
        # M-step
        t = np.mean(Ez, axis =1)
        for j in range(K):
            sumEz = np.sum(Ez[j])
            mu[j] = np.sum(np.diag(Ez[j]) @ X, axis = 0) / sumEz
            covar[j] = np.zeros((d, d))
            temp = X - mu[j]
            for i in range(n):
                covar[j] += Ez[j, i] * temp[i][:, None] @ temp[i][None, :]
            covar[j] /= sumEz
        diff = np.abs(muold - mu)
        muold = mu.copy()
        diff = np.sum(diff)
        if show:
            print('diff:', diff)
        if show and d < 4:
            plt.clf()
            plot(X, mu)
            plt.pause(.01)
    if show and d < 4:
        if d == 1:
            plt.figure()
            plot(X, mu)
            xlim = plt.gca().get_xlim()
            xstep = np.arange(xlim[0], xlim[1], .01)
            gmmplot(mu, covar, t, xstep, color = 'g')
        if d == 2:
            plt.figure()
            ax = Axes3D(plt.gcf())
            ax.scatter(X[:, 0], X[:, 1], np.zeros(len(X)), '.')
            ax.scatter(mu[:, 0], mu[:, 1], np.zeros(len(mu)), 'or')
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xstep = np.arange(xlim[0], xlim[1], .1)
            ystep = np.arange(ylim[0], ylim[1], .1)
            gmmplot3(mu, covar, t, xstep = xstep, ystep = ystep, alpha = 0.2, ax = ax)
        # d > 2 cannot plot pdf 
        print('done')
        plt.show() 
    # Distances
    D = np.zeros((K, len(X)))
    for k in range(K):
        D[k] = np.sum((X - mu[k]) ** 2, axis = 1)
    idx = np.argmin(D, axis = 0)
    return idx, mu
    
if __name__ == '__main__':
    K = 3
    N = 500 
    d = 2 
    X = np.empty((0, d))
    s = [0.1, 0.3, 0.15]
    m = [[2, .2, 1], [.2, 1.4, 2], [1.1, .6, 1]]
    for k in range(len(s)):
        X = np.vstack((X, s[k] * np.random.randn(N // len(s), d) + m[k][:d]))

    cluster, centroids = GMM_EM(X, K, show = True)