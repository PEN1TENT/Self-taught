# Decision Tree (ID3)
import os
import sys

from pandas.core.generic import NDFrameT
from sqlalchemy.util.langhelpers import FastIntFlag
from tables.leaf import NPByteArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "DecisionTree_module", "cp.5-dt-DecisionTree.py"
)
DecisionTree_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DecisionTree_module)
DecisionTree = DecisionTree_module.DecisionTree

import numpy as np


class ID3(DecisionTree):
    def __init__(self, X, T, F):
        super().__init__([], [], [])
        X = np.array(X)  # support list
        T = np.array(T)  # support list
        self.attr = F
        F = np.array(F)
        self.attr_id = []
        self.build_tree(X, T, F)

    @staticmethod
    def entropy_S(T):
        target = np.unique(T)
        N = len(T)
        E = 0
        for t in target:
            n = np.sum(T == t)
            P = n / N
            temp = np.log2(P) if P > 0 else 0
            E -= P * temp
        return E

    @staticmethod
    def best_split(X, T, Es, F):
        target = np.unique(T)
        num, num_attr = X.shape
        Gain = np.full(num_attr, Es)
        for j in range(num_attr):
            Fea = np.unique(X[:, j])
            for f in Fea:
                idx = X[:, j] == f
                n = np.sum(idx)
                E = 0
                for t in target:
                    m = np.sum(T[idx] == t)
                    P = m / n
                    temp = np.log2(P) if P > 0 else 0
                    E -= P * temp
                Gain[j] -= n / num * E

        Fi = np.argmax(Gain)
        return Fi, F[Fi]

    def build_tree(self, X, T, F, parent=-1, branch=None):
        Es = self.entropy_S(T)
        self.parent += [parent + 1]
        
        if branch is not None:
            self.branch += [branch]
        
        if Es == 0:
            if len(T) > 0:
                self.node += [T[0]] 
                self.attr_id += [None]
                
        elif len(F) == 0:
            if len(T) == 0:
                return 
            vals, counts = np.unique(T, return_counts=True)
            majority_class = vals[np.argmax(counts)]
            self.node += [majority_class]
            self.attr_id += [None]
            
        else:
            Fi, Fs = self.best_split(X, T, Es, F)
            self.node += [Fs]
            self.attr_id += [i for i, f in enumerate(self.attr) if f == Fs]
            parent = len(self.parent) - 1
            
            idf = F != Fs
            Vs = np.unique(X[:, Fi])
            for v in Vs:
                idx = X[:, Fi] == v
                if np.any(idx):
                    self.build_tree(X[idx][:, idf], T[idx], F[idf], parent, v)

    def predict(self, X):
        X = np.array(X)
        Y = []
        for i, x in enumerate(X):
            node = 0
            while True:
                # Find children
                children = [j for j, p in enumerate(self.parent) if p == node + 1]
                if len(children) == 0:
                    Y.append(self.node[node])
                    break
                not_found = True
                for c in children:
                    if self.branch[c - 1] == x[self.attr_id[node]]:
                        node = c
                        not_found = False
                        break
                if not_found:
                    Y.append(None)  # Out of values
                    break
        return np.array(Y)
