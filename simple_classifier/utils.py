import numpy as np


def _check_y(y):
    _classes = np.unique(y)
    if len(_classes) != 2:
        assert()


def _convert_y(y):
    _classes = np.unique(y)
    if np.any(np.isin(_classes, [-1, 1], invert=True)):
        return 2 * (y == _classes.max()) - 1
    return y
