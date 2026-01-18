# Car dataset loader
import pandas as pd
import os 
import numpy as np

def load(path = './dataset/car.csv', return_attr_names = False, split_train_test = None):
    if os.path.isfile(path):
        car = pd.read_csv(path)
    else:
        url =  'http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
        car = pd.read_csv(url, header = None)
        car.to_csv(path, index = False)
        
    attr_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    
    X = car.iloc[:, :6].values
    T = car.iloc[:, -1].values
    F = np.array(attr_names)
    if split_train_test:
        classes = np.unique(T)
        itrain = np.empty((0, ), dtype = int)
        itest = np.empty((0, ), dtype = int)
        for i in classes:
            idx = np.where(T == i)[0]
            split = int(len(idx) * split_train_test)
            itrain = np.concatenate((itrain, idx[:split]))
            itest = np.concatenate((itest, idx[split:]))
        return X[itrain], T[itrain], X[itest], T[itest]
    
    if return_attr_names:
        return X, T, F
    return X, T