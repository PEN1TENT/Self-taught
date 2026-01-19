# Self Organizing Maps
import numpy as np
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def plot(X, mu, xy):
    # 1D
    if X.shape[1] == 1:
        plt.plot(X, np.zeros(len(X)), '.')
        plt.plot(mu, np.zeros(len(mu)), 'or', mfc = 'none')
    # 2D
    if X.shape[1] == 2:
        plt.plot(X[:, 0], X[:, 1], '.')
        plt.plot(mu[:, 0], mu[:, 1], 'or', mfc = 'none')
        for i, x in enumerate(xy):
            for j, y in enumerate(xy):
                if np.sum(np.abs(x - y)) == 1:
                    plt.plot(mu[[i, j], 0], mu[[i, j], 1], 'g')
    # 3D
    if X.shape[1] > 2:
        ax = Axes3D(plt.gcf())
        ax.plot(X[:, 0], X[:, 1], X[:, 2], '.')
        ax.plot(mu[:, 0], mu[:, 1], mu[:, 2], 'or', mfc = 'none')
        for i, x in enumerate(xy):
            for j, y in enumerate(xy):
                if np.sum(np.abs(x - y)) == 1:
                    ax.plot(mu[[i, j], 0], mu[[i, j], 1], mu[[i, j], 2], 'g')

def SOM(X, K, decay = 'linear', LR_0 = 0.5, LR_T = 0.1, Sigma_0 = 3, Sigma_T = 0.1, n_iter = 1e3, show = False, return_xy = False):
    if isinstance(K, int):
        K = [K]
    # decay = 'linear' or 'nonlinear'
    n, d = X.shape
    
    # Initial SOM 
    if decay == 'nonlinear':
        radius = np.max(K) / 2
        timeScale = n_iter / np.log(radius) 
    
    # Over dimension 
    if len(K) > d:
        K = K[:d]
        
    N_node = np.prod(K)
    # Initial means by randomly selecting from dataset 
    if len(K) == 1:
        # 1D Grid 
        mu = X[np.random.choice(len(X), K[0], replace = False)]
        mu = np.sort(mu)
        xy = np.hstack((np.arange(N_node).reshape(N_node, 1), np.zeros((N_node, 1)))) # grid coordinates 
    elif len(K) > 1:
        # 2D to nD Grid 
        xy = np.unravel_index(np.arange(N_node), K)
        xy = np.vstack(xy).T
        mu = xy.copy()
        mu = np.hstack((mu, np.random.rand(N_node, d-len(K)))) # keep dimension 
    
    for t in range(int(n_iter)):
        # select a sample 
        v = X[np.random.choice(len(X))]
        
        # Calculate distances
        D = np.sum((mu - v) ** 2, axis = 1)
        
        # Best Matching Unit (BMU)
        BMU = np.argmin(D)
        
        # Radius & Learning rate 
        if decay == 'linear':
            Sigma2 = (Sigma_0 + (t / n_iter) * (Sigma_T - Sigma_0)) ** 2 
            LR = LR_0 + (t / n_iter) * (LR_T - LR_0)
        if decay == 'nonlinear':
            Sigma2 = (radius * np.exp(-t / timeScale)) ** 2
            LR = LR_T * np.exp(-t / n_iter)
        
        # Update weights
        distBMU = np.sum((xy - xy[BMU]) ** 2, axis = 1)
        NB = np.exp(-distBMU / (2 * Sigma2)) # Neighbourhood function 
        mu += (LR * NB * (v - mu).T).T # use transpose for broadcast trick 
        
        if show:
            print('%d: LR = %f BMU = %d' % (t, LR, BMU))
            plt.clf()
            plot(X, mu, xy)
            plt.pause(1e-3)
    
    # Distances 
    K = np.prod(K)
    D = np.zeros((K, len(X)))
    for k in range(K):
        D[k] = np.sum((X - mu[k]) ** 2, axis = 1)
    idx = np.argmin(D, axis = 0)
    if return_xy:
        return idx, mu, xy
    else:
        return idx, mu
        
if __name__ == '__main__':
    K = 3
    N = 500
    d = 2
    X = np.empty((0, d))
    s = [0.1, 0.3, 0.15]
    m = [[2, .2, 1], [.2, 1.4, 2], [1.1, .6, 1]]
    for k in range(len(s)):
        X = np.vstack((X, s[k] * np.random.rand(N // len(s), d) + m[k][:d]))
    
    cluster, centroids, xy = SOM(X, K, show = False, return_xy = True)
    plot(X, centroids, xy)
    plt.show()