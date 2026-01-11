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
    
classes = np.unique(y)
svm = []
label_pair = []
IDX = []
for i, label_i in enumerate(classes[:-1]):
    for j, label_j in enumerate(classes[i + 1:]):
        svm.append(SVM(C = 1000, kernel = 'rbf')) 
        y_ = np.zeros(y.shape)
        idx = (y == label_i) | (y == label_j)
        IDX.append(idx)
        y_[y == label_i] = 1
        y_[y == label_j] = -1
        svm[-1].train(X[idx], y_[idx]) 
        label_pair.append([label_i, label_j])
        
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
    plt.plot(X[IDX[k]][svm[k].iSV, 0], X[IDX[k]][svm[k].iSV, 1], 'ks', mfc = 'none')
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
    plt.plot(X[IDX[k]][svm[k].iSV, 0], X[IDX[k]][svm[k].iSV, 1], 'ks', mfc = 'none') 

# Hard margin I 
class_cmap = LinearSegmentedColormap.from_list('class_cmap', [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
plt.figure()
ax = plt.gca()
FX = np.array(FX)

# Find classes 
C = np.zeros((fx.shape[0], fx.shape[1], K), int)
for k in range(len(svm)):
    C[:, :, k][FX[k] >= 0] = label_pair[k][0]
    C[:, :, k][FX[k] < 0] = label_pair[k][1]
# Majority voting 
C_ = np.zeros((fx.shape[0], fx.shape[1]), int)
for m in range(C.shape[0]):
    for n in range(C.shape[1]):
        C_[m, n] = np.argmax(np.bincount(C[m, n]))
        
C = C_ 
C = np.rot90(C, -1).T
for k in range(len(svm)):
    # Plot data 
    for i, label in enumerate(classes):
        idx = y == label 
        ax.plot(X[idx, 0], X[idx, 1], colors[i])
ax.imshow(C, extent = [xlim[0], xlim[1], ylim[0], ylim[1]], alpha = 0.2, cmap = class_cmap)

# Soft  margin 
plt.figure()
ax = plt.gca()
soft = np.zeros((K, fx.shape[0], fx.shape[1]), float)
for k in range(len(svm)):
    soft[int(label_pair[k][0]), :, :][FX[k] >= 0] += FX[k][FX[k] >= 0]
    soft[int(label_pair[k][1]), :, :][FX[k] < 0] += np.abs(FX[k][FX[k] < 0])
# Normalization 
Cmax = np.max(soft, axis = 0)
Cmin = np.min(soft, axis = 0)
C = (soft - Cmin) / (Cmax - Cmin)
soft_margin = np.zeros((fx.shape[0], fx.shape[1], 3), float)
for k in range(len(svm)):
    # Plot data 
    for i, label in enumerate(classes):
        idx = y == label
        ax.plot(X[idx, 0], X[idx, 1], colors[i])
        soft_margin[:, :, k] = np.rot90(C[k], -1).T 
ax.imshow(soft_margin, extent = [xlim[0], xlim[1], ylim[0], ylim[1]], alpha = 0.2)

# Hard margin II 
C = np.argmax(soft_margin, axis = 2)
hard_margin = np.zeros((soft_margin.shape[0], soft_margin.shape[1], 3), np.uint8)
for k in range(K):
    hard_margin[:, :, k][C == k] = 255 
plt.figure()
ax = plt.gca()
for k in range(len(svm)):
    # Plot data
    for i, label in enumerate(classes):
        idx = y == label 
        ax.plot(X[idx, 0], X[idx, 1], colors[i])
    
ax.imshow(hard_margin, extent = [xlim[0], xlim[1], ylim[0], ylim[1]], alpha = 0.2)
plt.show()