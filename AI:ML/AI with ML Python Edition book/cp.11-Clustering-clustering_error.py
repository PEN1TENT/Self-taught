# Clustering error evaluations
import numpy as np
import warnings

def centroid_error(idx, centroids, X, target):
    K = len(centroids)
    IDX = -np.ones(idx.shape)
    # Find actual centroids 
    e = 0
    c = []
    classess, Y = np.unique(target, return_inverse = True)
    for i in range(K):
        c.append(np.mean(X[Y == i], axis = 0))
        d = np.sqrt(np.sum((centroids - c[01]) ** 2, axis = 1))
        id = np.argmin(d)
        e += d[id]
    e /= K
    if np.any(IDX == -1):
        warnings.warn('Some clusters failed')
    # Find accuracy rate 
    a = np.sum(IDX == Y) / len(Y) * 100
    return e, a