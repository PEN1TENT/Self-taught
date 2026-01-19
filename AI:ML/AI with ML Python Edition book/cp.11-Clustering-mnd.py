# Multicariate Normal Distribution 
import numpy as np
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 

def mnd(x, mu, C):
    temp = x - mu 
    if np.linalg.det(C) < 1e-17: # near sigular 
        gamma = 1e-10 
        Cinv = np.linalg.inv(C.T @ C + gamma * np.eye(C.shape[0])) @ C.T 
    else:
        Cinv = np.linalg.inv(C)
    p = np.exp(-temp @ Cinv @ temp.T / 2)
    p *= 1 / np.sqrt((2 * np.pi) ** x.shape[1] * np.linalg.det(C))
    return np.diag(p)
    
def mndplot3(mu, C, xstep = np.arange(-4, 4.1, .1), ystep = np.arange(-4, 4.1, .1), edgecolor = np.random.rand(3), color = np.random.rand(3), alpha = 1):
    x1, x2 = np.meshgrid(xstep, ystep)
    Z = np.vstack((x1.flatten(), x2.flatten())).T 
    G = mnd(Z, mu, C)
    G = G.reshape(x1.shape)
    ax = Axes3D(plt.gcf())
    ax.plot_surface(x1, x2, G, color = color, alpha = alpha, edgecolor = edgecolor)

if __name__ == '__main__':
    mu = np.zeros((2))
    C = np.eye(2)
    mndplot3(mu, C)
    plt.show()