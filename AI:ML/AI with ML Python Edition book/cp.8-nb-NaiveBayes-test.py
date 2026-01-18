import importlib.util
import sys

spec = importlib.util.spec_from_file_location("naiveBayes1", "cp.8-nb-naiveBayes.py")
naiveBayes_module = importlib.util.module_from_spec(spec)
sys.modules["naiveBayes1_module"] = naiveBayes_module
spec.loader.exec_module(naiveBayes_module)
NaiveBayes = naiveBayes_module.NaiveBayes

import numpy as np

spec1 = importlib.util.spec_from_file_location(
    "spam_dataset_module", "cp.8-nb-spam_dataset.py"
)
spam_dataset_module = importlib.util.module_from_spec(spec1)
sys.modules["spam_dataset_module"] = spam_dataset_module
spec1.loader.exec_module(spam_dataset_module)

Xtrain, Ytrain, Xtest, Ytest = spam_dataset_module.load(split_train_test = 0.8)
nb = NaiveBayes(Xtrain, Ytrain, is_text = True)
Ztest = nb.test(Xtest)
rate = np.sum(Ztest == Ytest) / len(Ytest) * 100
print('Accuracy rate [Spam]', rate)