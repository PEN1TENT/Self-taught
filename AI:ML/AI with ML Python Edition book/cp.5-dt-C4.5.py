import numpy as np
from collections import Counter

class ID3(DecisionTree):
    def __init__(self, X, T, F):
        super().__init__([], [], [])
        self.X = np.array(X)
        self.T = np.array(T)
        self.attr = np.array(F)
        self.attr_id = []
        self.thresholds = {} # Store thresholds for continuous splits
        self.build_tree(self.X, self.T, self.attr)

    @staticmethod
    def entropy_S(T):
        if len(T) == 0: return 0
        _, counts = np.unique(T, return_counts=True)
        probs = counts / len(T)
        return -np.sum(probs * np.log2(probs))

    def best_split(self, X, T, Es, F):
        num_samples, num_attr = X.shape
        best_gain = -1
        split_info = {"idx": 0, "name": F[0], "is_continuous": False, "threshold": None}

        for j in range(num_attr):
            column = X[:, j]
            # Check if column is numeric
            is_numeric = np.issubdtype(column.dtype, np.number)
            
            if is_numeric:
                # CONTINUOUS SPLIT LOGIC
                sorted_indices = np.argsort(column)
                sorted_col = column[sorted_indices]
                sorted_T = T[sorted_indices]
                
                # Test midpoints between adjacent unique values
                unique_vals = np.unique(sorted_col)
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
                
                for thresh in thresholds:
                    left_idx = column <= thresh
                    right_idx = ~left_idx
                    
                    if not np.any(left_idx) or not np.any(right_idx):
                        continue
                        
                    e_left = self.entropy_S(T[left_idx])
                    e_right = self.entropy_S(T[right_idx])
                    
                    current_gain = Es - (np.sum(left_idx)/num_samples * e_left + 
                                         np.sum(right_idx)/num_samples * e_right)
                    
                    if current_gain > best_gain:
                        best_gain = current_gain
                        split_info = {"idx": j, "name": F[j], "is_continuous": True, "threshold": thresh}
            else:
                # CATEGORICAL SPLIT LOGIC (Existing)
                unique_f = np.unique(column)
                weighted_e = 0
                for f in unique_f:
                    idx = column == f
                    weighted_e += (np.sum(idx)/num_samples) * self.entropy_S(T[idx])
                
                current_gain = Es - weighted_e
                if current_gain > best_gain:
                    best_gain = current_gain
                    split_info = {"idx": j, "name": F[j], "is_continuous": False, "threshold": None}

        return split_info

    def build_tree(self, X, T, F, parent=-1, branch=None):
        Es = self.entropy_S(T)
        self.parent += [parent + 1]
        if branch is not None: self.branch += [branch]

        # Base Cases
        if Es == 0 or len(F) == 0 or len(T) <= 1:
            val = Counter(T).most_common(1)[0][0] if len(T) > 0 else None
            self.node += [val]
            self.attr_id += [None]
            return

        split = self.best_split(X, T, Es, F)
        Fi = split["idx"]
        Fs = split["name"]
        
        self.node += [Fs]
        # Map back to original attribute index
        self.attr_id += [list(self.attr).index(Fs)]
        curr_node_idx = len(self.parent) - 1

        if split["is_continuous"]:
            thresh = split["threshold"]
            self.thresholds[curr_node_idx] = thresh
            
            # Binary split for continuous
            for op in [f"<= {thresh:.2f}", f"> {thresh:.2f}"]:
                idx = (X[:, Fi] <= thresh) if "<=" in op else (X[:, Fi] > thresh)
                if np.any(idx):
                    # Note: F is NOT modified for continuous attributes
                    self.build_tree(X[idx], T[idx], F, curr_node_idx, op)
        else:
            # Multi-way split for categorical
            idf = F != Fs
            Vs = np.unique(X[:, Fi])
            for v in Vs:
                idx = X[:, Fi] == v
                if np.any(idx):
                    self.build_tree(X[idx][:, idf], T[idx], F[idf], curr_node_idx, v)