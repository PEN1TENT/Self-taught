# Principal Component Analysis (PCA)
import numpy as np 
from matplotlib import pyplot as plt

def PCA(X, d = 0.95, show = False):
    mu = np.mean(X, axis = 0)
    X -= mu # Centralization 
    # Check SSS problem
    N, dim = X.shape 
    SSS = N < dim 
    if SSS:
        Sx = 1 / (len(X) - 1) * X @ X.T 
    else:
        Sx = 1 / (len(X) - 1) * X.T @ X
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
    if SSS:
        A = X.T @ A
        A = A / np.linalg.norm(A, axis = 0) / (N - 1)
        #A = A / np.sqrt(D[:d] * (N - 1)) # NaN when eigenvalue is 0 
    return A, mu, X @ A
    