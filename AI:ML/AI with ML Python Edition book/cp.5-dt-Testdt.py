import os
import sys

from pandas.core.generic import NDFrameT
from sqlalchemy.util.langhelpers import FastIntFlag
from tables.leaf import NPByteArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "playTennis_dataset_module", "cp.5-dt-playTennis_dataset.py"
)
playTennis_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(playTennis_dataset)

specs = importlib.util.spec_from_file_location("ID3_module", "cp.5-dt-ID3.py")
ID3_module = importlib.util.module_from_spec(specs)
specs.loader.exec_module(ID3_module)
ID3 = ID3_module.ID3

import numpy as np

X, T, F = playTennis_dataset.load(return_attr_name=True)
tree = ID3(X, T, F)
print(tree.parent)
print(tree.node)
print(tree.attr_id)
print(tree.branch)
tree.show()

Ztest = tree.predict(X)
rate = np.sum(T == Ztest) / len(T) * 100
print("Accuracy rate (evaluate by training set) =", rate, "%")

Xnew = [["Rain", "Hot", "High", "Strong"], ["Cloudy", "Hot", "High", "Strong"]]
Znew = tree.predict(Xnew)
print(Znew)
