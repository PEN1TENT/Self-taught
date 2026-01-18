# Classify by k-NN
import numpy as np
from matplotlib import pyplot as plt 
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("PCA", "cp.10-DR-PCA.py")
PCA_module = importlib.util.module_from_spec(spec)
sys.modules["PCA1_module"] = PCA_module
spec.loader.exec_module(PCA_module)
PCA = PCA_module.PCA

spec2 = importlib.util.spec_from_file_location("kNN_module", "cp.4-kNN-kNN.py")
kNN_module = importlib.util.module_from_spec(spec2)
sys.modules["kNN_module"] = kNN_module
spec2.loader.exec_module(kNN_module)
kNN = kNN_module.kNN

spec3 = importlib.util.spec_from_file_location(
    "ORL_dataset_module", "cp.10-DR-ORL_dataset.py"
)
ORL_dataset_module = importlib.util.module_from_spec(spec3)
sys.modules["ORL_dataset_module"] = ORL_dataset_module
spec3.loader.exec_module(ORL_dataset_module)

Xtrain, Ltrain, Xtest, Ltest = ORL_dataset_module.load(split_train_test = 0.5)
n_PC = np.min(Xtrain.shape)
A, mu, Ytrain = PCA(Xtrain, n_PC) # get all PCs
Ytest = (Xtest - mu) @ A # project test set 
rate = [] 
for i in range(n_PC):
    Ztest = kNN(Ytrain[:, :i + 1], Ltrain, Ytest[:, :i + 1])
    rate.append(np.sum(Ztest == Ltest) / len(Ztest) * 100)
    
max_rate = max(rate)
print('Best accuracy:', max_rate, 'PCA', rate.index(max_rate) + 1)

plt.plot(rate)
plt.xlabel('Dimension')
plt.ylabel('Accuracy (%)')
plt.show()