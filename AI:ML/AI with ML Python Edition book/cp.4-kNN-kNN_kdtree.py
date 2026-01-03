# kNN_kdtree
import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np
spec = importlib.util.spec_from_file_location("kdtree_module", "cp.4-kNN-kdtree.py")
kdtree_module = importlib.util.module_from_spec(spec)
sys.modules["kdtree_module"] = kdtree_module
spec.loader.exec_module(kdtree_module)
kdtree = kdtree_module.kdtree
node2tree = kdtree_module.node2tree


class Currentbest:
    d = []

    def __init__(self, dim):
        self.X = np.empty((0, dim + 1))  # add 1 dim for label


def kNN_kdtree(node, X, k=1):
    label = []
    for i in range(X.shape[0]):
        currentbest = Currentbest(X.shape[1])
        currentbest = NN_kdtree2(X[i, :], node, currentbest, k)
        (values, counts) = np.unique(currentbest.X[:, -1], return_counts=True)
        ind = np.argmax(counts)
        label += [values[ind]]
    return label


def NN_kdtree2(X, node, currentbest, k):
    if node is None or node.med is None:
        return currentbest
    if node.right is None and node.left is None:  # leaf
        d = np.sum((X - node.med[:, :-1]) ** 2)
        if len(currentbest.d) == k and currentbest.d[-1] > d:
            currentbest.d = np.append(currentbest.d, d)
            IDX = np.argsort(currentbest.d)
            currentbest.d = currentbest.d[IDX]
            currentbest.X = np.vstack((currentbest.X, node.med))
            currentbest.X = currentbest.X[IDX]
            currentbest.d = currentbest.d[:-1]
            currentbest.X = currentbest.X[:-1]

        elif len(currentbest.d) < k:
            currentbest.d = np.append(currentbest.d, d)
            IDX = np.argsort(currentbest.d)
            currentbest.d = currentbest.d[IDX]
            currentbest.X = np.vstack((currentbest.X, node.med))
            currentbest.X = currentbest.X[IDX]
    else:
        if X[node.axis] <= node.med:
            currentbest = NN_kdtree2(X, node.left, currentbest, k)
            if (
                len(currentbest.d) < k
                or (X[node.axis] - node.med) ** 2 <= currentbest.d[-1]
            ):
                currentbest = NN_kdtree2(X, node.right, currentbest, k)
        else:
            currentbest = NN_kdtree2(X, node.right, currentbest, k)
            if (
                len(currentbest.d) < k
                or (X[node.axis] - node.med) ** 2 <= currentbest.d[-1]
            ):
                currentbest = NN_kdtree2(X, node.left, currentbest, k)

    return currentbest


if __name__ == "__main__":
    spec1 = importlib.util.spec_from_file_location("kNN_iris", "cp.4-kNN-kNN_iris.py")
    kNN_iris = importlib.util.module_from_spec(spec1)
    sys.modules["kNN_iris_module"] = kNN_iris
    spec1.loader.exec_module(kNN_iris)

    spec2 = importlib.util.spec_from_file_location(
        "iris_dataset", "cp.4-kNN-iris_dataset.py"
    )
    iris_dataset = importlib.util.module_from_spec(spec2)
    sys.modules["iris_dataset_module"] = iris_dataset
    spec2.loader.exec_module(iris_dataset)

    spec_kdtree = importlib.util.spec_from_file_location(
        "kdtree_module", "cp.4-kNN-kdtree.py"
    )
    kdtree_module = importlib.util.module_from_spec(spec_kdtree)
    sys.modules["kdtree_module"] = kdtree_module
    spec_kdtree.loader.exec_module(kdtree_module)

    spec_treeplot = importlib.util.spec_from_file_location(
        "treeplot_module", "cp.5-dt-treeplot.py"
    )
    treeplot_module = importlib.util.module_from_spec(spec_treeplot)
    sys.modules["treeplot_module"] = treeplot_module
    spec_treeplot.loader.exec_module(treeplot_module)

    kdtree = kdtree_module.kdtree
    node2tree = kdtree_module.node2tree
    treeplot = treeplot_module.treeplot
    plotdata = kNN_iris.plotdata

    Xtrain, Ytrain, Xtest, Ytest = iris_dataset.load_iris_dataset(split_train_test=0.5)

    node = kdtree(Xtrain, Ytrain)
    tree = node2tree(node)

    print(f"Number of nodes: {len(tree)}")
    print(f"Tree parent list: {tree}")

    treeplot(tree, show=True)

    rate = []
    plt.figure()
    K = range(1, len(Xtrain) + 1)
    for k in K:
        Ztest = kNN_kdtree(node, Xtest, k)

        plotdata(Xtrain, Ytrain, Xtest, Ytest, Ztest)
        plt.title("k = " + str(k))
        plt.draw
        plt.pause(0.1)
        plt.cla()
        accuracy = np.sum(Ytest == Ztest) / len(Ytest) * 100
        rate.append(accuracy)

    plt.figure()
    plt.plot(K, rate)
    plt.xlabel("k")
    plt.ylabel("Accuracy rate (%)")
    plt.show()
