# Principal Component Analysis (PCA)
import numpy as np 
import matplotlib.pyplot as plt 

def PCA(X, d = 0.95, show = False):
    mu = np.mean(X, axis = 0)
    X -= mu # Centralization 
    Sx = 1 / (len(X) - 1) * X.T @ X # Covariance matrix 
    D, A = np.linalg.eig(Sx) 
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
    return A, mu, X @ A 
    