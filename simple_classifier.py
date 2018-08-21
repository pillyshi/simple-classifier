import numpy as np
from sklearn.base import BaseEstimator
from utils import _check_y, _convert_y


class SimpleClassifier(BaseEstimator):

    def __init__(self, threshold=None, convert_y=True):
        self.threshold = threshold
        self.convert_y = convert_y

    def fit(self, X, y):
        _check_y(y)
        if self.convert_y:
            y = _convert_y(y)
        n, d = X.shape
        if self.threshold is None:
            self.threshold = X.mean(axis=0)
        Z = 2 * (X >= self.threshold) - 1
        self.w = Z.T.dot(y) / n

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def decision_function(self, X):
        Z = 2 * (X >= self.threshold) - 1
        return Z.dot(self.w)


class EnsembleRandomSimpleClassifier(BaseEstimator):

    def __init__(self, n_estimators=100, random_state=None, convert_y=True):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.convert_y = convert_y

    def fit(self, X, y):
        _check_y(y)
        if self.convert_y:
            y = _convert_y(y)
        rnd = np.random.RandomState(self.random_state)
        n, d = X.shape
        self.thresholds = X[rnd.permutation(n)[:self.n_estimators]]
        self.W = np.empty((self.n_estimators, d))
        for i in range(self.thresholds.shape[0]):
            Z = 2 * (X >= self.thresholds[i]) - 1
            self.W[i] = Z.T.dot(y) / n
        return self

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def decision_function(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(self.thresholds.shape[0]):
            Z = 2 * (X >= self.thresholds[i]) - 1
            y_pred += np.sign(Z.dot(self.W[i]))
        return y_pred
