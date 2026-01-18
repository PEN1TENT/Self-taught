import importlib.util
import sys

spec = importlib.util.spec_from_file_location("PCA1", "cp.10-DR-PCA1.py")
PCA1_module = importlib.util.module_from_spec(spec)
sys.modules["PCA1_module"] = PCA1_module
spec.loader.exec_module(PCA1_module)
PCA = PCA1_module.PCA

spec3 = importlib.util.spec_from_file_location(
    "ORL_dataset_module", "cp.10-DR-ORL_dataset.py"
)
ORL_dataset_module = importlib.util.module_from_spec(spec3)
sys.modules["ORL_dataset_module"] = ORL_dataset_module
spec3.loader.exec_module(ORL_dataset_module)

X, L = ORL_dataset_module.load()
A, mu, Y = PCA(X, show = True)