import importlib.util
import sys

spec = importlib.util.spec_from_file_location("PCA1", "cp.10-DR-PCA1.py")
PCA1_module = importlib.util.module_from_spec(spec)
sys.modules["PCA1_module"] = PCA1_module
spec.loader.exec_module(PCA1_module)
PCA = PCA1_module.PCA

spec3 = importlib.util.spec_from_file_location(
    "iris_dataset_module", "cp.4-kNN-iris_dataset.py"
)
iris_dataset_module = importlib.util.module_from_spec(spec3)
sys.modules["iris_dataset_module"] = iris_dataset_module
spec3.loader.exec_module(iris_dataset_module)

X, L = iris_dataset_module.load_iris_dataset()

A, mu, Y = PCA(X, 2, show = True) # Reduce to 2D

print('Projection matrix:')
print(A)

from matplotlib import pyplot as plt 
color = {'Iris-setosa' : 'r', 'Iris-versicolor' : 'g', 'Iris-virginica' : 'b', }
marker = {'Iris-setosa' : 'o', 'Iris-versicolor' : 'x', 'Iris-virginica' : 's', }
for i in range(len(Y)):
    plt.plot(Y[i][0], Y[i][1], marker[L[i]], c = color[L[i]], markersize = 3)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
plt.show()