# Car dataset
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "car_dataset_module", "cp.5-dt-car_dataset.py"
)
car_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(car_dataset)

specs = importlib.util.spec_from_file_location("ID3_module", "cp.5-dt-ID3.py")
ID3_module = importlib.util.module_from_spec(specs)
specs.loader.exec_module(ID3_module)
ID3 = ID3_module.ID3

X, T, F = car_dataset.load(return_attr_names=True)
tree = ID3(X, T, F)
tree.show()
