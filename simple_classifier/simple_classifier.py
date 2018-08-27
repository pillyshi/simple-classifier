import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import _ovr_decision_function
from .utils import _check_y, _convert_y


class SimpleBinaryClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, threshold=None, convert_y=True):
        self.threshold = threshold
        self.convert_y = convert_y

    def fit(self, X, y, sample_weight=1):
        self.classes_ = np.unique(y)
        _check_y(y)
        if self.convert_y:
            y = _convert_y(y)
        n, d = X.shape
        if self.threshold is None:
            self.threshold = X.mean(axis=0)
        Z = 2 * (X >= self.threshold) - 1
        Z = Z * np.c_[sample_weight]
        self.w_ = Z.T.dot(y) / n

    def predict(self, X):
        return np.int32(self.decision_function(X) >= 0)

    def decision_function(self, X):
        Z = 2 * (X >= self.threshold) - 1
        return Z.dot(self.w_)


class SimpleRandomBinaryClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, sigma=0.01, convert_y=True, random_state=None):
        self.sigma = sigma
        self.convert_y = convert_y
        self.random_state = random_state

    def fit(self, X, y, sample_weight=1):
        self.classes_ = np.unique(y)
        _check_y(y)
        if self.convert_y:
            y = _convert_y(y)
        n, d = X.shape
        rnd = np.random.RandomState(self.random_state)
        mean = X.mean(axis=0)
        self.threshold_ = rnd.multivariate_normal(mean, np.identity(d) * self.sigma)
        Z = 2 * (X >= self.threshold_) - 1
        Z = Z * np.c_[sample_weight]
        self.w_ = Z.T.dot(y) / n
        return self

    def predict(self, X):
        indices = np.int32(self.decision_function(X) >= 0)
        return self.classes_[indices]

    def decision_function(self, X):
        Z = 2 * (X >= self.threshold_) - 1
        return Z.dot(self.w_)

# TODO: implement SimpleClassifier which can handle multiclass classification
# TODO: implement SimpleRandomClassifier which can handle multiclass classification
