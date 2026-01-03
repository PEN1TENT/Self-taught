# LSH Performance
import importlib.util
import sys
import numpy as np

# Load LSH module
lsh_spec = importlib.util.spec_from_file_location("cp.4-kNN-LSH", "cp.4-kNN-LSH.py")
lsh_module = importlib.util.module_from_spec(lsh_spec)
sys.modules["cp.4-kNN-LSH"] = lsh_module
lsh_spec.loader.exec_module(lsh_module)
LSH = lsh_module.LSH

# Load kNN module
knn_spec = importlib.util.spec_from_file_location("cp.4-kNN-kNN", "cp.4-kNN-kNN.py")
knn_module = importlib.util.module_from_spec(knn_spec)
sys.modules["cp.4-kNN-kNN"] = knn_module
knn_spec.loader.exec_module(knn_module)
kNN = knn_module.kNN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def genData(dim = 100000, nperClass = 50, nClass = 3):
    Xtrain = []
    Xtest = []
    Ytrain = []
    Ytest = []
    for i in range(nClass):
        mu = (i + 1) * np.random.randn(dim)
        Xtrain.append(np.random.randn(nperClass, dim) + mu)
        Xtest.append(np.random.randn(nperClass, dim) + mu)
        Ytrain.append(i*np.ones(nperClass, dtype = int))
        Ytest.append(i*np.ones(nperClass, dtype = int))
    return np.vstack(Xtrain), np.vstack(Xtest), np.hstack(Ytrain), np.hstack(Ytest)
    
if __name__=='__main__':
    nClass = 3
    nperClass = 50
    DIM = range(1, 4002, 1000)
    k = 1
    repeat = 50
    tLSH = np.zeros((len(DIM), repeat))
    rateLSH = np.zeros((len(DIM), repeat))
    tkNN = np.zeros((len(DIM), repeat))
    ratekNN = np.zeros((len(DIM), repeat))
    for i, dim in enumerate(DIM):
        Xtrain, Xtest, Ytrain, Ytest = genData(dim, nperClass, nClass)
        for I in range(repeat):
            # LSH
            t1 = time.time()
            YtestLSH = LSH(Xtrain, Ytrain, Xtest, k)
            tLSH[i, I] = time.time() - t1
            rateLSH[i, I] = np.sum(Ytest == YtestLSH) / len(Ytest) * 100
            print('{0:d}: LSH = {1:.02f} s {2: .02f} %'.format(dim, tLSH[i, I], rateLSH[i, I]), end='\t')
            
            # kNN
            t1 = time.time()
            YtestkNN = kNN(Xtrain, Ytrain, Xtest, k)
            tkNN[i, I] = time.time() - t1
            ratekNN[i, I] = np.sum(Ytest == YtestkNN) / len(Ytest) * 100
            print(f'kNN = {0:.02f} s {1:.02f} %'.format(tkNN[i, I], ratekNN[i, I]))
            
    plt.figure()
    plt.plot(DIM, np.mean(rateLSH, axis = 1), 'b')
    plt.plot(DIM, np.mean(ratekNN, axis = 1), '--r')
    plt.xlabel('Dimension')
    plt.ylabel('Accuracy (%)')
    plt.legend(['LSH', 'kNN'])
    
    plt.figure()
    plt.plot(DIM, np.mean(tLSH, axis = 1), 'b')
    plt.plot(DIM, np.mean(tkNN, axis = 1), '--r')
    plt.xlabel('Dimension')
    plt.ylabel('Processing time (s)')
    plt.legend(['LSH', 'kNN'])
    plt.show()