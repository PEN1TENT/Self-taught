# Naive Bayes classifier 
import numpy as np 

def log_gaussian(x, mu, sigma):
    return np.log(1. / (np.sqrt(2. * np.pi) * sigma)) + (-np.power((x - mu) / sigma, 2.))
    
class NaiveBayes:
    def __init__(self, X, Y, is_continuous = None, is_text = False):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.targets, counts = np.unique(self.Y, return_counts = True)
        self.Nv = len(self.targets) # The number of targets (classes)
        self.Fv = counts.astype(float)
        self.Pv = self.Fv / len(self.Y) # P(V_j)
        self.is_text = is_text 
        if self.is_text:
            self.train_text()
        else:
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
    def train_text(self):
        self.vocab = set()
        self.Fav = {}
        self.n = {}
        for j, t in enumerate(self.targets):
            self.Fav[t] = {}
            self.n[t] = np.sum(np.vectorize(len)(self.X[self.Y == t]))
        for i, x in enumerate(self.X):
            counts = {}
            for w in x:
                counts[w] = counts.get(w, 0) + 1
            for word, count in counts.items():
                if word not in self.vocab:
                    self.vocab.add(word)
                if word not in self.Fav[self.Y[i]]:
                    self.Fav[self.Y[i]][word] = 0
                self.Fav[self.Y[i]][word] += count
                
    def test(self, X):
        Z = []
        for i, x in enumerate(X): 
            P = np.zeros(len(self.targets))
            if self.is_text:
                for word in set(x):
                    if word not in self.vocab:
                        continue
                    for j, t in enumerate(self.targets):
                        P[j] += np.log((self.Fav[t].get(word, 0.0) + 1) / (self.n[t] + len(self.vocab)))
                for j, t in enumerate(self.targets):
                    P[j] += np.log(self.Pv[j])
            else:
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