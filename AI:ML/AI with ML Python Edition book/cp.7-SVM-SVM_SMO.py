# SVM with Sequential Minimal Optimization (SMO)
from re import L
import numpy as np

import importlib.util
import sys

spec = importlib.util.spec_from_file_location("Kernel_module", "cp.7-SVM-Kernel.py")
Kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Kernel)
Kernel = Kernel.Kernel

class SVM:
    def __init__(self, C = 10, kernel = 'rbf', degree = 3, gamma = 1, a = 1, b = 1):
        self.C = C
        self.kernel = kernel 
        self.degree = degree 
        self.gamma = gamma 
        self.a = a
        self.b = b 
    
    def train(self, X, y):
        self.y = y
        self.SMO(X, y)
        
    def examineExample(self, i2, y2):
        alpha2 = self.alpha[i2]
        if 0 < alpha2 < self.C:
            E2 = self.error[i2]
        else:
            E2 = self.SVMout(i2) - y2 
        r2 = E2 * y2 
        if (r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0):
            # First Choise
            # Find the point with the largest difference |E1 - E2|
            imin = np.argmin(self.error)
            Emin = self.error[imin]
            imax = np.argmax(self.error)
            Emax = self.error[imax]
            if abs(E2 - Emin) > abs(E2 - Emax):
                if self.takeStep(imin, i2):
                    return 1
            else:
                if self.takeStep(imax, i2):
                    return 1
            
            # Non Bound
            vio_alpha = np.where((self.alpha < self.C) & (self.alpha > 0))[0]
            if len(vio_alpha) > 0:
                np.random.shuffle(vio_alpha)
                for i1 in vio_alpha:
                    if self.takeStep(i1, i2):
                        return 1
            
            # Bound
            vio_alpha = np.where((self.alpha == self.C) | (self.alpha == 0))[0]
            if len(vio_alpha) > 0:
                np.random.shuffle(vio_alpha)
                for i1 in vio_alpha:
                    if self.takeStep(i1, i2):
                        return 1
            
            # Entire set
            all_i1 = np.arange(len(self.alpha))
            np.delete(all_i1, i2)
            np.random.shuffle(all_i1)
            for i1 in all_i1:
                if self.takeStep(i1, i2):
                    return 1
                    
        return 0
    
    def SVMout(self, i):
        return (self.y * self.alpha) @ self.K[:, i] - self.bias
        
    def predict(self, X):
        return (self.y * self.alpha) @ Kernel(self.sv, X, kernel = self.kernel).K - self.bias
    
    def SMO(self, X, y):
        N = len(X)
        self.alpha = np.zeros(N)
        self.bias = 0
        self.error = np.zeros(N)
        self.tol = 1e-3
        self.K = Kernel(X, kernel = self.kernel, degree = self.degree, gamma = self.gamma, a = self.a, b =self.b).K
        self.eps = 1e-3
        numChanged = 0
        examineAll = True
        while numChanged or examineAll:
            numChanged = False
            if examineAll:
                for I in range(N):
                    numChanged += self.examineExample(I, y[I])
                else:
                    vio_alpha = np.where((self.alpha < self.C) & (self.alpha > 0))[0]
                    for I in vio_alpha:
                        numChanged += self.examineExample(I, y[I])
            if examineAll:
                examineAll = False
            elif numChanged == 0:
                examineAll = True
    
        # Output
        iSV = np.where(self.alpha > .5)[0]
        self.sv = X[iSV]
        self.y = y[iSV]
        self.alpha = self.alpha[iSV]
        self.iSV = iSV
    
    def takeStep(self, i1, i2):
        i1 = int(i1)
        i2 = int(i2)
        if i1 == i2:
            return False
        alpha1 = self.alpha[i1]
        alpha2 = self.alpha[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]
        
        if 0 < alpha1 < self.C:
            E1 = self.error[i1]
        else: 
            E1 = self.SVMout(i1) - y1
        
        if 0 < alpha2 < self.C:
            E2 = self.error[i2]
        else:
            E2 = self.SVMout(i2) - y2
        
        s = y1 * y2
        
        if y1 == y2:
            L = max([0, alpha1 + alpha2 - self.C])
            H = min([self.C, alpha1 + alpha2])
        else:
            L = max([0, alpha2 - alpha1])
            H = min([self.C, self.C - alpha1 + alpha2])
        
        if L == H: # No line segment
            return False
        
        k11 = self.K[i1, i1]
        k12 = self.K[i2, i1]
        k22 = self.K[i2, i2]
        eta = k11 + k22 - 2 * k12
        if eta > 0:
            a2 = alpha2 + y2 * (E1 - E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            f1 = y1 * (E1 + self.bias) - alpha1 * k11 - s * alpha2 * k12
            f2 = y2 * (E2 + self.bias) - alpha2 * k22 - s * alpha1 * k12 
            L1 = alpha1 + s * (alpha2 - L)
            H1 = alpha1 + s * (alpha2 - H)
            Lobj = L1 * f1 + L * f2 + 0.5 * L1 * L1 * k11 + 0.5 * L + L * k22 + s * L * L1 * k12
            Hobj = H1 * f1 + H * f2 + 0.5 * H1 * H1 * k11 + 0.5 * H + H * k22 + s * H * H1 * k12
            
            if Lobj < Hobj - self.eps:
                a2 = L
            elif Lobj > Hobj + self.eps:
                a2 = H
            else:
                a2 = alpha2
        if abs(a2 - alpha2) < self.eps * (a2 + alpha2 + self.eps):
            return False
        a1 = alpha1 + s * (alpha2 - a2)
        
        # Update threshold to reflect cahnge in Lagrange multipliers
        b1 = E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 + self.bias
        b2 = E2 +y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + self.bias
        if 0 < a1 < self.C:
            bnew = b1 
        elif 0 < a2 < self.C:
            bnew = b2 
        else:
            bnew = (b1 + b2) / 2
        self.bias = bnew
        
        # Update weight vector to reflect change in a1 & a2, if SVM is linear 
        if self.kernel == 'linear':
            if not hasattr(self, 'w'):
                self.w = y1 * (a1 - alpha1) * X[i1] + y2 * (a2 - alpha2) * X[i2]
            else:
                self.w += y1 * (a1 - alpha1) * X[i1] + y2 * (a2 - alpha2) * X[i2]
            
        # Update error cache using new Lagrange multipliers
        self.alpha[i1] = a1 
        self.alpha[i2] = a2 
        
        svmoutall = self.SVMout(slice(0, len(self.y)))
        self.error = svmoutall - self.y 
        self.rate = np.sum(np.sign(svmoutall) == self.y) / len(self.y) * 100 
        
        return True