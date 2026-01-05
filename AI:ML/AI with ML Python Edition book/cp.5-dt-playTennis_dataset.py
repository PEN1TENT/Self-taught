# Play Tennis Dataset Generator
import numpy as np


def load(return_attr_name=False, split_train_test=None):
    X = [
        ["Sunny", "Hot", "High", "Weak"],
        ["Sunny", "Hot", "High", "Strong"],
        ["Overcast", "Hot", "High", "Weak"],
        ["Rain", "Mild", "High", "Weak"],
        ["Rain", "Cool", "Normal", "Weak"],
        ["Rain", "Cool", "Normal", "Strong"],
        ["Overcast", "Cool", "Normal", "Strong"],
        ["Sunny", "Mild", "High", "Weak"],
        ["Sunny", "Cool", "Normal", "Weak"],
        ["Rain", "Mild", "Noraml", "Weak"],
        ["Sunny", "Mild", "Normal", "Strong"],
        ["Overcast", "Mild", "High", "Strong"],
        ["Overcast", "Hot", "Normal", "Weak"],
        ["Rain", "Mild", "High", "Strong"],
    ]
    F = ["Outlook", "Temp", "Humidity", "Wind"]
    T = [
        "No",
        "No",
        "Yes",
        "Yes",
        "Yes",
        "No",
        "Yes",
        "No",
        "Yes",
        "Yes",
        "Yes",
        "Yes",
        "Yes",
        "No",
    ]
    X = np.array(X)
    T = np.array(T)
    F = np.array(F)
    if split_train_test:
        classes = np.unique(T)
        itrain = np.empty((0,), dtype=int)
        itest = np.empty((0,), dtype=int)
        for i in classes:
            idx = np.where(T == i)[0]
            split = int(len(idx) * split_train_test)
            itrain = np.concatenate((itrain, idx[:split]))
            itest = np.concatenate((itest, idx[split:]))
        return X[itrain], T[itrain], X[itest], T[itest]

    if return_attr_name:
        return X, T, F
    return X, T
