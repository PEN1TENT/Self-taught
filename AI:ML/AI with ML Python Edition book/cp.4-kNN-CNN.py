# Condensed Nearest Neighbor
import numpy as np

def CNN(Xtrain, Ytrain):
    nsample = Xtrain.shape[0]
    # Distance
    Dtrain = []
    for x in Xtrain:
        Dtrain.append(np.sqrt(np.sum((Xtrain - x)**2, axis = 1)))
    Dtrain = np.array(Dtrain)
    
    # Border ratio
    a = []
    for i in range(nsample):
        # y = the closet sample from x with different class from x
        yidx = np.argmin(Dtrain[i, Ytrain != Ytrain[i]])
        # xd = the closet sample from y with same class as x
        xdidx = np.where(Ytrain == Ytrain[i])[0]
        xdidx = np.delete(xdidx, np.where(xdidx == np.array(i)))
        xdidx = np.delete(xdidx, np.where(xdidx == yidx))
        xdidx = np.argmin(Dtrain[yidx, xdidx])
        a.append(Dtrain[yidx, xdidx]/Dtrain[i, yidx])
        
    order = np.argsort(-np.array(a))
    # Prototypes
    Prototypes = order[0]
    i = 0
    while len(order) > 0 and i < len(order):
        idx = np.argsort(Dtrain[order[i], Prototypes])
        for j in idx:
            if Ytrain[order[i]] != Ytrain[j]:
                Prototypes = np.append(Prototypes, order[i])
                order = np.delete(order, i)
                break
        i += 1
        
    return Prototypes
    
if __name__ == '__main__':
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
    
    spec3 = importlib.util.spec_from_file_location("kNN_iris_module", "cp.4-kNN-kNN_iris.py")
    kNN_iris_module = importlib.util.module_from_spec(spec3)
    sys.modules["kNN_iris_module"] = kNN_iris_module
    spec3.loader.exec_module(kNN_iris_module)
    plotdata = kNN_iris_module.plotdata
    
    import matplotlib.pyplot as plt
    
    Xtrain, Ytrain, Xtest, Ytest = iris_dataset.load(split_train_test = 0.5)
    Prototypes = CNN(Xtrain, Ytrain)
    
    rate = []
    
    plt.figure(1)
    K = range(1, len(Prototypes) + 1)
    for k in K:
        Ztest = kNN(Xtrain[Prototypes], Ytrain[Prototypes], Xtest, k)
        
        plotdata(Xtrain[Prototypes], Ytrain[Prototypes], Xtest, Ytest, Ztest)
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
    print(rate)
    
    plt.figure(3)
    maxrate = max(rate)
    print(maxrate)
    k = rate.index(maxrate) + 1
    Ztest = kNN(Xtrain[Prototypes], Ytrain[Prototypes], Xtest, k)
    plotdata(Xtrain[Prototypes], Ytrain[Prototypes], Xtest, Ytest, Ztest)
    plt.title('k = ' + str(k))
    
    plt.figure(4)
    plt.subplot(121)
    plotdata(Xtrain[Prototypes], Ytrain[Prototypes])
    plt.title('Prototypes')
    
    plt.subplot(122)
    plotdata(Xtrain, Ytrain)
    plt.title('All training samples')
    plt.show()