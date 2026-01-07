# Feedforward Backpropagation Neural Network (iris dataset)
import numpy as np 
from matplotlib import pyplot as plt

import importlib.util
import sys
spec = importlib.util.spec_from_file_location(
    "FFBP_sigmoid_module", "cp.6-ANN-FFBP_sigmoid.py"
)
FFBP_sigmoid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FFBP_sigmoid)
FFBP = FFBP_sigmoid.FFBP
spec2 = importlib.util.spec_from_file_location("iris_dataset_module", "cp.4-kNN-iris_dataset.py")
iris_dataset_module = importlib.util.module_from_spec(spec2)
sys.modules["iris_dataset_module"] = iris_dataset_module
spec2.loader.exec_module(iris_dataset_module)

def label2onehot(target):
    targets = sorted(list(set(target)))
    idx = [targets.index(t) for t in target]
    return np.eye(len(targets), dtype = 'uint8')[idx], targets
    
if __name__ == '__main__':
    Xtrain, Ytrain, Xtest, Ytest = iris_dataset_module.load_iris_dataset(split_train_test = 0.5)
    
    T, class_name = label2onehot(Ytrain)
    
    hidden = [5]
    
    net = FFBP(Xtrain, T, hidden, lr = 0.001, alpha = 0.8, eps = 0.01)
    
    plt.plot(net.error)
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.show()
    
    Ztest = net.forward(Xtest)
    Ztest = [class_name[i] for i in np.argmax(Ztest, axis = 1)]
    rate = np.sum(Ytest == Ztest) / len(Ytest) * 100
    print('Accuracy rate =', rate, '%')