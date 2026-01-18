# ORL Dataset loader 
import os 
from matplotlib.image import imread
import numpy as np

def load(path = r'./att_faces', return_vector = True, normalize = True, split_train_test = None):
    X = []
    Y = []
    for i in os.listdir(path):
        if os.path.isdir(os.path.join(path, i)):
            for j in os.listdir(os.path.join(path, i)):
                if 'pgm' in j:
                    fname = os.path.join(path, i ,j)
                    img = imread(fname)
                    if return_vector:
                          img = img.flatten()
                    if normalize:
                        img = img / 255 
                    X.append(img)
                    Y.append(i)
    X = np.array(X)
    Y = np.array(Y)
    if split_train_test:
        classes = np.unique(Y)
        itrain = np.empty((0,), dtype = int)
        itest = np.empty((0,), dtype = int)
        for i in classes:
            idx = np.where(Y == i)[0]
            split = int(len(idx) * split_train_test) # 50:50
            itrain = np.concatenate((itrain, idx[:split]))
            itest = np.concatenate((itest, idx[split:]))
        return X[itrain], Y[itrain], X[itest], Y[itest]
    return X, Y