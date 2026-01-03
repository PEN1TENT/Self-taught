# k-d tree
import numpy as np

class Node:
    med = None
    left = None
    right = None
    axis = -1


def kdtree(X, label):
    X, d = np.unique(X, axis=0, return_index=True)
    X = np.hstack((X, label[d].reshape((X.shape[0], 1))))
    node = kdtree2(X)
    return node


def kdtree2(X):
    node = Node()
    if len(X) == 0:
        return None

    if X.shape[0] == 1:
        node.med = X
        return node

    node.axis = np.argmax(np.max(X[:, :-1], axis=0) - np.min(X[:, :-1], axis=0))
    med = X.shape[0] // 2 - 1

    if med == -1:
        return med

    idx = np.argsort(X[:, node.axis])
    X = X[idx]
    node.med = (X[med, node.axis] + X[med + 1, node.axis]) / 2
    node.left = kdtree2(X[: med + 1])
    node.right = kdtree2(X[med + 1 :])

    return node


def node2tree(node):
    T = []

    def buildtree(node, parent):
        T.append(parent)
        idx = len(T)
        if node.left is not None:
            buildtree(node.left, idx)
        if node.right is not None:
            buildtree(node.right, idx)

    buildtree(node, 0)
    return T
