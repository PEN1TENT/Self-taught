# Gaussian Mixture Model (GMM)
import numpy as np
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("mnd", "cp.11-Clustering-mnd.py")
mnd_module = importlib.util.module_from_spec(spec)
sys.modules["mnd_module"] = mnd_module
spec.loader.exec_module(mnd_module)
mnd = mnd_module.mnd

from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 

def gmm(x, mu, C, w = None):
    p = np.zeros(len(x))
    if w is None:
        return mnd(x, mu, C)
    for k in range(len(w)):
        p += w[k] * mnd(x, mu[k], C[k])
    return p

def pdf(x, mu, sigma):
    return (1. / (np.sqrt(2. * np.pi) * sigma)) * np.exp(-((x - mu) / sigma) ** 2 / 2)

def mixpdf(x, mu, sigma, w = None):
    if w is None:
        return pdf(x, mu, sigma)
    else:
        p = np.zeros(len(x))
        for k in range(len(w)):
            p += w[k] * pdf(x, mu[k], flatten(), sigma[k].flatten())
        return p

def gmmplot(mu, C, w, xstep = np.arange(-4, 4.01, .01), color = np.random.rand(3), linestyle = '-'):
    G = mixpdf(xstep, mu, C, w)
    plt.plot(xstep, G, color = color, linestyle = linestyle)
    
def gmmplot_sep(mu, C, w, xstep = np.arange(-4, 4.01, .01), color = None, linestyle = ':'):
    # Plot each mixture 
    if color is None:
        color = np.random.rand(len(w), 3)
    for k in range(len(w)):
        G = mixpdf(xstep, [mu[k]], [C[k]], [w[k]])
        plt.plot(xstep, G, color = color[k], linestyle = linestyle)

def gengrid(xstep, ystep):
    x1, x2 = np.meshgrid(xstep, ystep)
    Z = np.vstack((x1.flatten(), x2.flatten())).T
    return x1, x2, Z 

def gmmplot3(mu, C, w, xstep = np.arange(-4, 4.1, .1), ystep = np.arange(-4, 4.1, .1), edgecolor = np.random.rand(3), color = np.random.rand(3), alpha = 1, ax = None):
    x1, x2, Z = gengrid(xstep, ystep)
    G = gmm(Z, mu, C, w)
    G = G.reshape(x1.shape)
    if ax is None:
        ax = Axes3D(plt.gcf())
    ax.plot_surface(x1, x2, G, color = color, alpha = alpha, edgecolor = edgecolor)
    
def gmmplot3_sep(mu, C, w, xstep = np.arange(-4, 4.1, .1), ystep = np.arange(-4, 4.1, .1), edgecolor = None, ax = None):
    # Plot each mixture 
    x1, x2, Z = gengrid(xstep, ystep)
    if ax is None:
        ax = Axes3D(plt.gcf())
    if edgecolor is None:
        edgecolor = np.random.rand(len(w), 3)
    for k in range(len(w)):
        G = gmm(Z, [mu[k]], [C[k]], [w[k]])
        G = G.reshape(x1.shape)
        ax.plot_surface(x1, x2, G, alpha = 0, edgecolor = edgecolor[k])
    
if __name__ == '__main__':
    M = 3 
    d = 2
    mu = np.random.rand(M, d) * 6 - 3
    w = np.random.rand(M)
    C = []
    for m in range(M):
        C.append(np.diag(np.random.rand(d)))
    
    if d == 1:
        gmmplot(mu, C, w)
        plt.figure()
        gmmplot_sep(mu, C, w, color = ['r', 'g', 'b'])
        plt.show()
        
    if d == 2:
        gmmplot3(mu, C, w)
        plt.figure()
        gmmplot3_sep(mu, C, w, edgecolor = ['r', 'g', 'b'])
        plt.show()