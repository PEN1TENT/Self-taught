# Random Projection on ORL
import numpy as np
from matplotlib import pyplot as plt 
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("RP", "cp.10-DR-RP.py")
RP_module = importlib.util.module_from_spec(spec)
sys.modules["RP_module"] = RP_module
spec.loader.exec_module(RP_module)
GRP = RP_module.GRP
SRP = RP_module.SRP

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

I = 100
d = 400 # np.max(Xtrain.shape)
RATE_GRP = []
RATE_SRP = []
for j in range(I):
    print(j)
    A_grp, mu_grp, Ytrain_grp = GRP(Xtrain, d)
    A_srp, mu_srp, Ytrain_srp = SRP(Xtrain, d)
    Ytest_grp = (Xtest - mu_grp) @ A_grp # project test set
    Ytest_srp = (Xtest - mu_srp) @ A_srp # project test set
    rate_grp = []
    rate_srp = []
    for i in range(d):
        Ztest_grp = kNN(Ytrain_grp[:, :i + 1], Ltrain, Ytest_grp[:, :i + 1])
        Ztest_srp = kNN(Ytrain_srp[:, :i + 1], Ltrain, Ytest_srp[:, :i + 1])
    RATE_GRP.append(rate_grp)
    RATE_SRP.append(rate_srp)
    
RATE_GRP = np.array(RATE_GRP).mean(axis = 0)
RATE_SRP = np.array(RATE_SRP).mean(axis = 0)
d = RATE_GRP.argmax()
print('GRP best average accuracy:', RATE_GRP[d], 'd:', d + 1)

d = RATE_SRP.argmax()
print('SRP best average accuracy:', RATE_SRP[d], 'd:', d + 1)

plt.plot(RATE_GRP)
plt.plot(RATE_SRP)
plt.legend(['GRP', 'SRP'])
plt.xlabel('Dimension')
plt.ylabel('Accuracy (%)')
plt.show()