# LSH
import numpy as np


def LSH(Xtrain, Ytrain, Xtest, k=1, d=32):
    # d i number of bits per signature
    Class = np.unique(Ytrain)
    nClass = len(Class)
    randv = np.random.randn(d, Xtrain.shape[1])
    LSHtrain = LSHsignature(Xtrain, randv)
    LSHtest = LSHsignature(Xtest, randv)
    # kNN with LSH
    Ytest = np.zeros(Xtrain.shape[0], dtype=int)
    D = np.zeros(Xtrain.shape[0], dtype=int)
    for j in range(Xtest.shape[0]):
        for i in range(Xtrain.shape[0]):
            D[i] = bin(LSHtrain[i] ^ LSHtest[j]).count("1")
        idx = np.argsort(D)
        c, f = np.unique(Ytrain[idx[:k]], return_counts=True)
        iClass = c[np.argmax(f)]
        Ytest[j] = Class[iClass]
    return Ytest


def LSHsignature(X, randv):
    res = np.zeros(X.shape[0], dtype=np.int64)
    for j in range(X.shape[0]):
        for i in range(randv.shape[0]):
            res[j] = res[j] << 1
            if np.dot(X[j], randv[i]) >= 0:
                res[j] = res[j] | 1

    return res
