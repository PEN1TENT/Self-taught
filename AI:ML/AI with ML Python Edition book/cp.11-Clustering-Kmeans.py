# K-means 
import numpy as np
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def plot(X, centroids = None):
    # Plot 1D, 2D and 3D spaces
    if X.shape[1] == 1:
        plt.plot(X.flatten(), np.zeros(len(X)), '.')
        if centroids is not None:
            plt.plot(centroids[:, 0].flatten(), np.zeros(len(centroids)), 'or', mfc = 'None')
    if X.shape[1] == 2:
        plt.plot(X[:, 0], X[:, 1], '.')
        if centroids is not None:
            plt.plot(centroids[:, 0], centroids[:, 1], 'or', mfc = None)
    if X.shape[1] == 3:
        ax = Axes3D(plt.gcf())
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], '.', s = 2)
        if centroids is not None:
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 'or', s = 50)
            
def Kmeans(X, K, th = 0, show = False):
    # Initial centroids by randomly selecting from dataset 
    centroids = X[np.random.choice(len(X), 3, replace = False)]
    i = 0
    while True:
        i += 1
        # Distance 
        D = np.zeros((K, len(X)))
        for k in range(K):
            D[k] = np.sum((X - centroids[k]) ** 2, axis = 1)
        # Update centroids
        idx = np.argmin(D, axis = 0)
        old_centroids = centroids.copy()
        for k in range(K):
            centroids[k] = np.mean(X[idx == k], axis = 0)
        # Display
        if show:
            print(i)
            print(centroids)
            if X.shape[1] <= 3:
                plt.clf()
                plot(X, centroids)
                plt.pause(.5)
        # Stopping criteria 
        if np.sum(np.abs(old_centroids - centroids)) <= th:
            if show:
                print('done')
            if show and X.shape[1] <= 3:
                plt.show()
            return idx, centroids
if __name__ == '__main__':
    K = 3
    N = 500
    d = 2
    X = np.empty((0, d))
    s = [0.1, 0.3, 0.15]
    m = [[2, .2, 1], [.2, 1.4, 2], [1.1, .6, 1]]
    for k in range(K):
        X = np.vstack((X, s[k] * np.random.randn(N // K, d) + m[k][:d]))
    
    cluster, centroids = Kmeans(X, K, show = True)