# Naive Bayes classifier
import numpy as np

def log_gaussian(x, mu, sigma):
    return np.log(1. / (np.sqrt(2. * np.pi) * sigma)) + (-np.power((x - mu) / sigma, 2.) / 2)

class NaiveBayes:
    def __init__(self, X, Y, is_continuous = None):
        self.X = np.array(X)
        self.Y = np.array(Y)
        
        self.targets, counts = np.unique(self.Y, return_counts = True)
        self.Nv = len(self.targets) # The number of targets (classes)
        self.Fv = counts.astype(float)
        self.Pv = self.Fv / len(self.Y) # P(V_j) 
        
        if is_continuous is None:
            self.is_continuous = [False] * X.shape[1]
        
        else:
            self.is_continuous = is_continuous 
        self.av = []
        for i in range(X.shape[1]):
            if self.is_continuous[i]:
                # For continuous values 
                self.av.append([])
            else:
                # For discontinuous values 
                self.av.append(np.unique(self.X[:, i])) # unique targets 
        self.train()
        
    def train(self):
        m = self.X.shape[1] # The number of attributes 
        self.Pav = {}
        for j, t in enumerate(self.targets):
            idx = self.Y == t 
            n = np.sum(idx)
            Pavt = []
            # Find P(a_i | v_j)
            for i in range(self.X.shape[1]):
                if self.is_continuous[i]:
                    temp = np.vectorize(float)(self.X[idx, i])
                    mu = np.mean(temp)
                    sigma = np.std(temp)
                    Pavt.append([mu, sigma])
                else:
                    P = {}
                    for av in self.av[i]:
                        P[av] = (np.sum(self.X[idx][:, i] == av) + 1) / (n + m) # avoid zero
                    Pavt.append(P) 
            self.Pav[t] = Pavt
 
    def test(self, X):
        Z = []
        for i, x in enumerate(X): 
            P = np.zeros(len(self.targets))
            for j, t in enumerate(self.targets):
                v = np.log(self.Pv[j])
                for w, a in enumerate(x):
                    if self.is_continuous[w]:
                        mu = self.Pav[t][w][0]
                        sigma = self.Pav[t][w][1]
                        if sigma != 0:
                            v += log_gaussian(float(a), mu, sigma)
                    else:
                        v += np.log(self.Pav[t][w][a])
                P[j] = v 
            Z.append(self.targets[P.argmax()])
        return Z