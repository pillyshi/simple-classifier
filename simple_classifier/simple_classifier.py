import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from .utils import _check_y, _convert_y


class SimpleClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, transformer='SRP', random_state=None, sigma=1, k=None):
        self.transformer = transformer
        self.random_state = random_state
        self.sigma = sigma
        self.k = k

    def fit(self, X, y, sample_weight=1):
        if not hasattr(self, 'classes_'):
            self.classes_ = np.unique(y)
        y = _convert_y(y, classes=self.classes_)
        n, d = X.shape
        if self.k is None:
            self.k = max(16, d)
        self._set_transformer(X)
        Z = self._transform(X)
        Z = Z * np.c_[sample_weight]
        self.w_ = Z.T.dot(y) / n
        self.n_samples_ = n
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=1):
        # if self.transformer != 'SRP':
        #     assert()
        if not hasattr(self, 'w_'):
            self.classes_ = classes
            self.fit(X, y)
            return self
        y = _convert_y(y, classes=self.classes_)
        n, d = X.shape
        if self.transformer == 'mean':
            mean = self.n_samples_ * self.mean_
            mean += X.sum(0)
            self.mean_ = mean / (self.n_samples_ + n)
        Z = self._transform(X)
        Z = Z * np.c_[sample_weight]
        w = self.n_samples_ * self.w_
        w += Z.T.dot(y)
        self.w_ = w / (self.n_samples_ + n)
        self.n_samples_ += n

    def predict(self, X):
        dec = self.decision_function(X)
        if len(self.classes_) > 2:
            indices = dec.argmax(axis=1)
        else:
            indices = np.int32(dec >= 0)
        return self.classes_[indices]

    def decision_function(self, X):
        Z = self._transform(X)
        return Z.dot(self.w_)

    def _set_transformer(self, X):
        d = X.shape[1]
        rnd = np.random.RandomState(self.random_state)
        if self.transformer == 'SRP':
            self.P_ = rnd.normal(0, 1, size=(d, self.k))
        elif self.transformer == 'mean':
            self.mean_ = X.mean(axis=0)
        elif self.transformer == 'mean-random':
            mean = X.mean(axis=0)
            self.threshold_ = rnd.multivariate_normal(mean, np.identity(d) * self.sigma)

    def _transform(self, X):
        if self.transformer == 'SRP':
            return np.sign(X.dot(self.P_))
        elif self.transformer == 'mean':
            return 2 * (X >= self.mean_) - 1
        elif self.transformer == 'mean-random':
            return 2 * (X >= self.threshold_) - 1
        assert()
