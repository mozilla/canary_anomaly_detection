# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from keras.metrics import mean_squared_error as mse_keras
from sklearn.metrics import mean_squared_error as mse_sklearn
from scipy.stats import wasserstein_distance, energy_distance


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


def log_bhattacharyya_dist(hist_1, hist_2):
    """
    Computes the log of 1 + bhattacharyya distance between two histograms
    """
    bc = np.sum(np.sqrt(np.abs(np.array(hist_1)) * np.abs(np.array(hist_2))))
    dist = -np.log(bc + 1e-8)
    return np.log(1 + dist) if not np.isnan(dist) else 0


def log_max_diff_dist(hist_1, hist_2):
    """
    Computes absolute difference between histograms by each bucket and takes
    log of 1 + maximum of the differences
    """
    diff = abs(np.array(hist_1) - np.array(hist_2))
    return np.log(1 + np.max(diff))


def log_sum_of_diff_dist(hist_1, hist_2):
    """
    Computes absolute difference between histograms by each bucket and takes
    log of 1 + sum of the differences
    """
    diff = abs(np.array(hist_1) - np.array(hist_2))
    return np.log(1 + np.sum(diff))


def log_wasserstein_dist(points_1, points_2):
    """
    Computes log of 1 + wasserstein_distance
    """
    return np.log(1 + wasserstein_distance(points_1, points_2))


def log_energy_dist(points_1, points_2):
    """
    Computes log of 1 + energy_distance
    """
    return np.log(1 + energy_distance(points_1, points_2))
