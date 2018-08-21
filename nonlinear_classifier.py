import numpy as np
from sklearn.base import BaseEstimator
from utils import _check_y, _convert_y


class NonlinearClassifier(BaseEstimator):

    def __init__(self, thresholds=None, sigma=1, n_components=100, random_state=None, convert_y=True):
        self.thresholds = thresholds
        self.sigma = sigma
        self.n_components = n_components
        self.random_state = random_state
        self.convert_y = convert_y

    def fit(self, X, y):
        _check_y(y)
        if self.convert_y:
            y = _convert_y(y)
        rnd = np.random.RandomState(self.random_state)

        Xce = X[rnd.permutation(len(X))[:self.n_components]]
        X2 = np.c_[np.sum(X**2, axis=1)]
        Xce2 = np.c_[np.sum(Xce**2, axis=1)]
        Phi = np.exp(- (X2 + Xce2.T - 2 * X.dot(Xce.T)) / (2 * self.sigma ** 2))
        mu = Phi.mean(axis=0)
        Z = 2 * (Phi >= mu) - 1
        n = Z.shape[0]
        self.w = Z.T.dot(y) / n
        self.Xce = Xce
        self.Xce2 = Xce2
        self.mu = mu

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def decision_function(self, X):
        X2 = np.c_[np.sum(X**2, axis=1)]
        Phi = np.exp(- (X2 + self.Xce2.T - 2 * X.dot(self.Xce.T)) / (2 * self.sigma ** 2))
        Z = 2 * (Phi >= self.mu) - 1
        return Z.dot(self.w)
