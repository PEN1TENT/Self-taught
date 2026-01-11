from matplotlib import pyplot as plt
import numpy as np
import time 

import importlib.util
import sys

from pyarrow.compute import RandomOptions

spec = importlib.util.spec_from_file_location("S_V_M_module", "cp.7-SVM-S_V_M.py")
S_V_M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S_V_M)
SVM = S_V_M.SVM

RATE = {'1-v-a': [], '1-v-1': [], 'DAG': []}
TRAINTIME = {'1-v-a': [], '1-v-1': [], 'DAG': []}
TESTTIME = {'1-v-a': [], '1-v-1': [], 'DAG': []}
for K in range(2, 50): # Number of classes 
    # Generate data
    print('K =', K)
    dim = 2 # Dimension 
    N = 150 # Number of samples 
    # data 
    X = np.random.rand(N // K, dim)
    for k in range(1, K):
        X = np.vstack((X, np.random.rand(N // K, dim) + 0.8 * k))
    # label 
    y = np.zeros((N // K))
    for k in range(1, K):
        y = np.hstack((y, np.full((N // K), k)))
    
    for method in ['1-v-a', '1-v-1', 'DAG']:
        svm = SVM(multiclass = method, verbose = False)
        t1 = time.time()
        svm.train(X, y)
        traintime = time.time() - t1
        TRAINTIME[method].append(traintime)
        print('Training time:', traintime, 's')
        t1 = time.time()
        z = svm.predict(X)
        testtime = time.time() - t1 
        TESTTIME[method].append(testtime)
        print('Test time:', testtime, 's')
        rate = np.sum(z == y) / len(y) * 100
        RATE[method].append(rate)
        print('Accuracy rate [Evaluated on training set]:', rate)
        
for method in ['1-v-a', '1-v-1', 'DAG']:
    plt.plot(RATE[method])
plt.title('Accuracy rate')
plt.legend(('1-v-a', '1-v-1', 'DAG'))
plt.xlabel('Number of classes')
plt.ylabel('%')
plt.figure()

for method in ['1-v-a', '1-v-1', 'DAG']:
    plt.plot(TRAINTIME[method])
plt.title('Training time (s)')
plt.legend(('1-v-a', '1-v-1', 'DAG'))
plt.xlabel('Number of classes')
plt.ylabel('seconds')
plt.figure()

for method in ['1-v-a', '1-v-1', 'DAG']:
    plt.plot(TESTTIME[method])
plt.title('Test time (s)')
plt.legend(('1-v-a', '1-v-1', 'DAG'))
plt.xlabel('Number of classes')
plt.ylabel('seconds')
plt.figure()

plt.show()