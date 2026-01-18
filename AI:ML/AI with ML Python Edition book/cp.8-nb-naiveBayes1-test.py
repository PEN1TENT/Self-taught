import importlib.util
import sys
from mailbox import NoSuchMailboxError

spec = importlib.util.spec_from_file_location("naiveBayes1", "cp.8-nb-naiveBayes1.py")
naiveBayes1_module = importlib.util.module_from_spec(spec)
sys.modules["naiveBayes1_module"] = naiveBayes1_module
spec.loader.exec_module(naiveBayes1_module)
NaiveBayes = naiveBayes1_module.NaiveBayes

import numpy as np

spec1 = importlib.util.spec_from_file_location(
    "playTennis_dataset_module", "cp.5-dt-playTennis_dataset.py"
)
playTennis_dataset_module = importlib.util.module_from_spec(spec1)
sys.modules["playTennis_dataset_module"] = playTennis_dataset_module
spec1.loader.exec_module(playTennis_dataset_module)
Xtrain, Ytrain, Xtest, Ytest = playTennis_dataset_module.load(split_train_test=0.8)
nb = NaiveBayes(Xtrain, Ytrain)
Ztest = nb.test(Xtest)
rate = np.sum(Ztest == Ytest) / len(Ytest) * 100
print("Accuracy rate [Play Tennis]", rate)

spec2 = importlib.util.spec_from_file_location(
    "car_dataset_module", "cp.5-dt-car_dataset.py"
)
car_dataset_module = importlib.util.module_from_spec(spec2)
sys.modules["car_dataset_module"] = car_dataset_module
spec2.loader.exec_module(car_dataset_module)
Xtrain, Ytrain, Xtest, Ytest = car_dataset_module.load(split_train_test=0.8)
nb = NaiveBayes(Xtrain, Ytrain)
Ztest = nb.test(Xtest)
rate = np.sum(Ztest == Ytest) / len(Ytest) * 100
print("Accuracy rate [Car]", rate)

spec3 = importlib.util.spec_from_file_location(
    "iris_dataset_module", "cp.4-kNN-iris_dataset.py"
)
iris_dataset_module = importlib.util.module_from_spec(spec3)
sys.modules["iris_dataset_module"] = iris_dataset_module
spec3.loader.exec_module(iris_dataset_module)
Xtrain, Ytrain, Xtest, Ytest = iris_dataset_module.load_iris_dataset(
    split_train_test=0.8
)
nb = NaiveBayes(Xtrain, Ytrain, [True, True, True, True])
Ztest = nb.test(Xtest)
rate = np.sum(Ztest == Ytest) / len(Ytest) * 100
print("Accuracy rate [Iris]", rate)
