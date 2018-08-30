import numpy as np
from keras.metrics import mean_squared_error as mse_keras
from sklearn.metrics import mean_squared_error as mse_sklearn


def mse_times_100_tf(y_true, y_pred):
    """
    Loss used for the model, `mean_squared_error` is a bit too small.
    Operates on tf arrays
    :param y_true: Ground truth array
    :param y_pred: Predicted array
    :return: `mean_squared_error` (from keras) loss times 100
    """
    return 100*mse_keras(y_true, y_pred)


def mse_times_100_np(y_true, y_pred):
    """
    Loss used for the model, `mean_squared_error` is a bit too small.
    Operates on np arrays
    :param y_true: Ground truth array
    :param y_pred: Predicted array
    :return: `mean_squared_error` (from sklearn) loss times 100
    """
    return 100*mse_sklearn(y_true, y_pred)


def bhattacharyya_dist(hist_1, hist_2):
    """
    Computes the bhattacharyya distance between two histograms
    """
    bc = np.sum(np.sqrt(np.abs(np.array(hist_1)) * np.abs(np.array(hist_2))))
    dist = -np.log(bc + 1e-8)
    return dist if not np.isnan(dist) else 0


def max_diff_dist(hist_1, hist_2):
    """
    Computes absolute difference between histograms by each bucket and takes
    maximum of the differences
    """
    diff = abs(np.array(hist_1) - np.array(hist_2))
    return np.max(diff)


def sum_of_diff_dist(hist_1, hist_2):
    """
    Computes absolute difference between histograms by each bucket and takes
    sum of the differences
    """
    diff = abs(np.array(hist_1) - np.array(hist_2))
    return np.sum(diff)
