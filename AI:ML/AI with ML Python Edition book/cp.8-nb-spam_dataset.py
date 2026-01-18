import pandas as pd
import numpy as np
import re

def load(path = './dataset/spam.csv', stopword_url = './dataset/stop-word-list.txt', split_train_test = None):
    tokenizer = re.compile("[a-zA-Z]{2.}").findall
    stopword = pd.read_csv(stopword_url, header = None)[0].values
    df = pd.read_csv(path, encoding = "ISO-8859-1")
    Y = df['v1'].values
    X = df['v2'].values
    for i, x in enumerate(X):
        X[i] = [j for j in tokenizer(x.lower()) if j not in stopword]
    X = np.array(X)
    Y = np.array(Y)
    if split_train_test:
        classes = np.unique(Y)
        itrain = np.empty((0, ), dtype = int)
        itest = np.empty((0, ), dtype = int)
        for i in classes:
            idx = np.where(Y == i)[0]
            split = int(len(idx) * split_train_test)
            itrain = np.concatenate((itrain, idx[:split]))
            itest = np.concatenate((itest, idx[split:]))
        return X[itrain], Y[itrain], X[itest], Y[itest]
    return X, Y