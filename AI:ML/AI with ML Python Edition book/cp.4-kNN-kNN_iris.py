import importlib.util
import sys
spec = importlib.util.spec_from_file_location("kNN_module", "cp.4-kNN-kNN.py")
kNN_module = importlib.util.module_from_spec(spec)
sys.modules["kNN_module"] = kNN_module
spec.loader.exec_module(kNN_module)
kNN = kNN_module.kNN
spec2 = importlib.util.spec_from_file_location("iris_dataset_module", "cp.4-kNN-iris_dataset.py")
iris_dataset_module = importlib.util.module_from_spec(spec2)
sys.modules["iris_dataset_module"] = iris_dataset_module
spec2.loader.exec_module(iris_dataset_module)

class iris_dataset:
    load = staticmethod(iris_dataset_module.load_iris_dataset)
import matplotlib.pyplot as plt
import numpy as np

def plotdata(Xtrain, Ytrain, Xtest = [], Ytest = [], Ztest = []):
    color = {'Iris-setosa': 'b',
            'Iris-versicolor': 'g',
            'Iris-virginica': 'r',}
    for i in range(len(Xtrain)):
        plt.plot(Xtrain[i][0], Xtrain[i][1], 'x', c = color[Ytrain[i]], mfc = 'none')
    for i in range(len(Xtest)):
        plt.plot(Xtest[i][0], Xtest[i][1], '-', c = 'none', mfc = color[Ytest[i]])
    for i in range(len(Ztest)):
        plt.plot(Xtest[i][0], Xtest[i][1], 'o', c = color[Ztest[i]], mfc = 'none')

if __name__ == '__main__':
    Xtrain, Ytrain, Xtest, Ytest = iris_dataset.load(split_train_test = 0.5)
    
    plt.figure(1)
    rate = []
    K = range(1, len(Xtrain) + 1)
    for k in K:
        Ztest = kNN(Xtrain, Ytrain, Xtest, k)
        plotdata(Xtrain, Ytrain, Xtest, Ytest, Ztest)
        plt.title('k = '+str(k))
        plt.draw()
        plt.pause(0.001)
        plt.cla()
        rate.append(np.sum(Ztest == Ytest)/len(Ytest) * 100)
        
    plt.figure(2)
    plt.plot(K, rate)
    plt.axis([0, 80, 30, 100])
    plt.xlabel('k')
    plt.ylabel('Accuracy rate (%)')
    plt.grid(True)
    plt.show()
    print(rate)
    
    plt.figure(3)
    k = rate.index(max(rate)) + 1
    Ztest = kNN(Xtrain, Ytrain, Xtest, k)
    plotdata(Xtrain, Ytrain, Xtest, Ytest, Ztest)
    plt.title('k = ' + str(k))
    plt.show()
    