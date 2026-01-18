# Linear Discriminant Analysis (LDA)
import numpy as np
from matplotlib import pyplot as plt 

def LDA(X, L, d, show = False):
    MU = np.mean(X, axis = 0)
    N, dim = X.shape 
    classes = np.unique(L)
    K = len(classes)
    Sb = np.zeros((dim, dim)) # Between-class Scatter Matrix 
    Sw = np.zeros((dim, dim)) # Within-class Scatter Matrix 
    mu = np.zeros((K, dim))
    for k, c in enumerate(classes):
        idx = L == c
        n = np.sum(idx)
        mu[k] = np.mean(X[idx], axis = 0)
        temp = mu[k] - MU
        Sb += n * (temp[:, None] @ temp[None, :])
        temp = X[idx] - mu[k]
        Sw += temp.T @ temp
    
    D, A = np.linalg.eig(np.linalg.inv(Sw) @ Sb)
    idx = np.argsort(D)[::-1] # Descending 
    D = D[idx]
    if show:
        plt.figure()
        plt.plot(D)
        plt.xlabel('Index')
        plt.ylabel('Eigenvalues')
        plt.show()
    if isinstance(d, float):
        th = d
        sumD = np.sum(D)
        for d in range(len(D)):
            if np.sum(D[:d + 1]) / sumD >= th:
                # Beware precision problem !!
                d += 1
                break
    A = A[:, idx[:d]]
    return A, MU, (X - MU) @ A
    