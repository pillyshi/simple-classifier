import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from .utils import _check_y, _convert_y


class Base(BaseEstimator, ClassifierMixin):

    def predict(self, X):
        dec = self.decision_function(X)
        if len(self.classes_) > 2:
            indices = dec.argmax(axis=1)
        else:
            indices = np.int32(dec >= 0)
        return self.classes_[indices]

    def decision_function(self, X):
        Z = 2 * (X >= self.threshold_) - 1
        return Z.dot(self.w_)
    

class SimpleClassifier(Base):

    def __init__(self, threshold=None):
        self.threshold_ = threshold

    def fit(self, X, y, sample_weight=1):
        self.classes_ = np.unique(y)
        y = _convert_y(y)
        n, d = X.shape
        if self.threshold_ is None:
            self.threshold_ = X.mean(axis=0)
        Z = 2 * (X >= self.threshold_) - 1
        Z = Z * np.c_[sample_weight]
        self.w_ = Z.T.dot(y) / n


class SimpleRandomClassifier(Base):

    def __init__(self, sigma=0.01, random_state=None):
        self.sigma = sigma
        self.random_state = random_state

    def fit(self, X, y, sample_weight=1):
        self.classes_ = np.unique(y)
        y = _convert_y(y)
        n, d = X.shape
        rnd = np.random.RandomState(self.random_state)
        mean = X.mean(axis=0)
        self.threshold_ = rnd.multivariate_normal(mean, np.identity(d) * self.sigma)
        Z = 2 * (X >= self.threshold_) - 1
        Z = Z * np.c_[sample_weight]
        self.w_ = Z.T.dot(y) / n
        return self
