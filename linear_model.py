import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime
import utils

# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares): # inherits the predict() function from LeastSquares
    def fit(self,X,y,z):
        # turn the weight vector to a diagonal matrix
        V = np.diag(z)
        self.w = solve(X.T@V@X, X.T@V@y)

class LinearModelGradient(LeastSquares):

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)

        # check the gradient
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient))
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self,w,X,y):
        n, d = X.shape

        f = 0
        g = 0
        for i in range(n):
            f = f + np.log(np.exp(w.T@X[i]-y[i]) + np.exp(y[i] - w.T@X[i]))
            g = g + X[i]*(np.exp(w.T@X[i]-y[i]) - np.exp(y[i] - w.T@X[i])) / (np.exp(w.T@X[i]-y[i]) + np.exp(y[i] - w.T@X[i]))
        return f,g


# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self,X,y):
        n, d = X.shape

        Z = np.insert(X, 0, np.ones(n), axis=1)
        self.v = solve(Z.T@Z, Z.T@y)

    def predict(self, X):
        n, d = X.shape

        Z = np.insert(X, 0, np.ones(n), axis=1)
        return Z@self.v

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        n, d = X.shape
        Z = np.ones((n, 1))
        for x in range(1, self.p + 1):
                Z = np.append(Z, np.power(X, x), axis=1)
        self.v = solve(Z.T@Z, Z.T@y)

    def predict(self, X):
        n, d = X.shape
        Z = np.ones((n, 1))
        for x in range(1, self.p + 1):
                Z = np.append(Z, np.power(X, x), axis=1)
        return Z@self.v

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        p = self.p
        Z = np.ones((X.shape[0],1))
        for i in range(1,p+1):
            Z = np.append(Z, np.power(X,i), axis=1)
        return Z
