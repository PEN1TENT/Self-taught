import importlib.util
from re import L
import sys

from matplotlib.lines import lineStyles

spec = importlib.util.spec_from_file_location("SVM_SMO_module", "cp.7-SVM-SVM_SMO.py")
SVM_SMO = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SVM_SMO)
SVM = SVM_SMO.SVM

from matplotlib.typing import LineStyleType
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm 
from matplotlib.colors import LinearSegmentedColormap

# Generate data
dim = 2 # Dimension 
N = 150 # Number of samples 
K = 3 # Number of classes 
# data 
X = np.random.rand(N // K, dim)
for k in range(1, K): 
    X = np.vstack((X, np.random.rand(N // K, dim) +  0.8 * k))
# labels
y = np.zeros((N // K))
for k in range(1, K):
    y = np.hstack((y, np.full((N // K), k)))
    
# SVM 
classes = np.unique(y)
svm = []
label_pair = []
for i in range(K - 1):
    for j in range(0, i + 1):
        I = j 
        J = K - i + j - 1
        print('SVM:', classes[I], 'vs', classes[J])
        svm.append(SVM(C = 1000, kernel = 'rbf'))
        y_ = np.zeros(y.shape)
        idx = (y == classes[I]) | (y == classes[J])
        y_[y != classes[I]] = 1
        y_[y != classes[J]] = -1
        svm[-1].train(X[idx], y_[idx])
        label_pair.append([I, J])
        
# Plot each hyperplane 
X1 = []
X2 = []
FX = []
colors = ['ro', 'g*', 'b.']
for k in range(len(svm)):
    plt.figure()
    # Plot data
    for i, label in enumerate(classes):
        idx = y == label
        plt.plot(X[idx, 0], X[idx, 1], colors[i])
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    x1, x2 = np.meshgrid(np.arange(xlim[0], xlim[1], 0.01), np.arange(ylim[0], ylim[1], 0.01))
    Z = np.vstack((x1.flatten(), x2.flatten())).T
    fx = svm[k].predict(Z)
    fx = fx.reshape(x1.shape)
    # Plot hyperplane 
    plt.contour(x1, x2, fx, 50, cmap = cm.coolwarm)
    cs = plt.contour(x1, x2, fx, levels = [-1, 0, 1], colors = ['k', 'k', 'k'])
    plt.clabel(cs, fmt = '%d', colors = 'k', fontsize = 14)
    # Plot support vector 
    plt.plot(X[svm[k].iSV, 0], X[svm[k].iSV, 1], 'ks', mfc = 'none')
    X1.append(x1)
    X2.append(x2)
    FX.append(fx)

# Plot all hyperplane 
plt.figure()
colors_cnt = ['r', 'g', 'b']
for k in range(len(svm)):
    # Plot data
    for i, label in enumerate(classes):
        idx = y == label
        plt.plot(X[idx, 0], X[idx, 1], colors[i])
    cs = plt.contour(X1[k], X2[k], FX[k], levels = [-1, 0, 1], colors = [colors_cnt[k], colors_cnt[k], colors_cnt[k]], lineStyles = 'dotted')
    plt.clabel(cs, fmt = '%d', colors = 'k', fontsize = 14)
    # Plot support vector 
    plt.plot(X[svm[k].iSV, 0], X[svm[k].iSV, 1], 'ks', mfc = 'none') 

# Classification (DAG)
C = []
for i in range(len(Z)):
    n = 0 
    for k in range(K - 1):
        fx = FX[n].ravel()[i]
        # end 
        if k == K - 2:
            if fx < 0:
                C.append(label_pair[n][0])
            else:
                C.append(label_pair[n][1])
        # move 
        if fx < 0:
            n += k + 1 # left
        else: 
            n += k + 2 # right

# Hard margin I
class_cmap = LinearSegmentedColormap.from_list('class_cmap', [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
plt.figure()
ax = plt.gca()
C = np.array(C)
C = C.reshape(FX[0]. shape)
C = np.rot90(C, -1).T
for k in range(len(svm)):
    # Plot data
    for i, label in enumerate(classes):
        idx = y == label 
        ax.plot(X[idx, 0], X[idx, 1], colors[i])
ax.imshow(C, extent = [xlim[0], xlim[1], ylim[0], ylim[1]], alpha = 0.2, cmap = class_cmap)
plt.show()

