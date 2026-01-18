# Random Projection 
import numpy as np

def GRP(X, d): # Gaussian Random Projection
    mu = np.mean(X, axis = 0)
    A = np.random.randn(X.shape[1], d)
    return A, mu, (X - mu) @ A 
    
def SRP(X, d): # Sparse Random Projection 
    mu = np.mean(X, axis = 0)
    A = np.random.randn(X.shape[1], d)
    
    hist, bin = np.histogram(A[:], bins = 100, density = True)
    p = hist * np.diff(bin)
    for i in range(len(p)):
        if np.sum(p[:i]) >= 1 / 6:
            break
            
    for j in range(-1, -len(p), -1):
        if np.sum(p[j:]) >= 1 / 6:
            break
    
    idx1 = (bin[0] <= A) & (A <= bin[i + 1])
    idx2 = (bin[i + 1] <= A) & (A <= bin[j - 2])
    idx3 = (bin[j - 2] <= A) & (A <= bin[-1])
    A[idx1] = 3 ** .5 
    A[idx2] = 0
    A[idx3] = -(3 ** .5)
    
    return A, mu, (X - mu) @ A