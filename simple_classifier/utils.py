import numpy as np
from sklearn.preprocessing import LabelBinarizer


def _check_y(y):
    _classes = np.unique(y)
    if len(_classes) != 2:
        assert()


def _convert_y(y, classes=None):
    if classes is None:
        classes = np.unique(y, axis=0)
    if len(classes) > 2:
        Y = np.c_[y] == classes
        return 2 * Y - 1
    return 2 * np.int32(y > classes.min()) - 1
