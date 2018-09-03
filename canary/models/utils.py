# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from collections import defaultdict

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from canary.generator.utils import X_metric_to_df, y_metric_to_df, buckets_to_points


def percentile(array, point):
    """
    Computes the value of empirical CDF for a given point, i.e. counts the values
    smaller than given point and divides by length of an array
    :param array: array of values from some distribution
    :param point: value to compute empirical CDF for
    """
    return np.mean([i <= point for i in array])


def plot_dists(X_dist, X_hist, y_hist=None, limits=(-0.5, 0.5), legend_names=None):
    """
    Plots the compare between the histogram data and the distances data
    :param X_dist: Data frame of one metric with the different distances as columns
    :param X_hist: Data frame of one metric with histograms as columns
    :param y_hist: The true anomalies
    :param limits: Limits for the distance plot. Used if the number of distance values
    differs from the number of histogram values.
    :param legend_names: Names for the legend for the distance plot. Used if X_dist
    has no column names.
    """
    plt.figure()
    # important magic numbers
    width_1 = 1.6
    width_2 = 2
    height = 1.5

    if y_hist is not None:
        width_2 = 2.5
    plt.title('Comparing changes in distances and anomalies')
    ax0 = plt.axes([0, height, width_1, height], label='0')
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.plot(X_dist)
    plt.xlim(limits[0], len(X_dist) - limits[1])
    plt.xlabel('Date')
    plt.ylabel('Distance value')
    if legend_names is None:
        legend_names = X_dist.columns
    plt.legend(legend_names)

    plt.axes([0, 0, width_2, height], label='1')
    df = X_metric_to_df(X_hist)
    sns.heatmap(df, cmap="YlGnBu")
    if y_hist is not None:
        y_df = y_metric_to_df(y_hist)
        sns.heatmap(df, mask=1 - y_df[0], cmap='YlOrRd')
    plt.xlabel('Date')
    plt.ylabel('Bucket')
    plt.subplots_adjust(hspace=.0)
    plt.show()


def plot_preds(X_df, y_true_df, y_pred_df):
    """
    Plots the the heatmap with true anomalies and the heatmap with predicted anmoalies
    :param X_df: Data frame of one metric with histograms as columns
    :param y_true_df: Data frame with true anomalies
    :param y_pred_df: Data frame with predicted anomalies
    """
    a4_dims = (15, 6)
    plt.subplots(figsize=a4_dims)
    sns.heatmap(X_df, cmap="YlGnBu")
    sns.heatmap(X_df, mask=1 - y_true_df[0], cmap='YlOrRd')
    plt.xlabel('Date')
    plt.ylabel('Bucket')
    plt.title('True')
    plt.show()

    a4_dims = (15, 6)
    plt.subplots(figsize=a4_dims)
    sns.heatmap(X_df, cmap="YlGnBu")
    sns.heatmap(X_df,
                mask=1 - y_pred_df.set_index(y_true_df.index)[0], cmap='YlOrRd')
    plt.xlabel('Date')
    plt.ylabel('Bucket')
    plt.title('Prediction')
    plt.show()


def calculate_dists(data, bucket_dists, points_dists):
    """
    Calculates the distances between histograms
    :param data: The whole X_dict
    :param bucket_dists: list of distance functions that should be calculated on bucket
    version of the histogram
    :param points_dists: list of distance functions that should be calculated on point
    version of the histogram
    :return: the dictionary of structure:
    {
        'METRIC_1': {
            'dist_1': [0.2, 0.5, 1.0],
            'dist_2': [0.2, 0.1, 3.0],
            'dist_3': [1.0, 0.6, 0.7],
            'dist_4': [0.2, 0.4, 0.1],
            'dist_5': [3.9, 2.7, 0.9],
        },
        ...
    }
    """
    dists = defaultdict(lambda: defaultdict(list))
    for metric, hist in data.items():
        dates = sorted(hist['data'].keys())
        date_pairs = [(dates[i], dates[i + 1]) for i in range(len(dates) - 1)]
        points = buckets_to_points(hist)
        for d1, d2 in date_pairs:
            day_dists = _calculate_one_day_dist(hist, points, d1, d2,
                                                bucket_dists, points_dists, metric)
            for dist in bucket_dists + points_dists:
                dists[metric][dist.__name__].append(day_dists[dist.__name__])
    return dists


def _calculate_one_day_dist(hist, points, date1, date2,
                            bucket_dists, points_dists, metric):
    """
    Calculates the distances for one day of data, see the docs for `calculate_dists`
    """
    dists_day = defaultdict(list)
    hist_1 = hist['data'][date1]
    hist_2 = hist['data'][date2]
    points_1 = points['data'][date1]
    points_2 = points['data'][date2]
    try:
        if len(hist_1) > len(hist_2):
            hist_2 += [0] * (len(hist_1) - len(hist_2))
        elif len(hist_2) > len(hist_1):
            hist_1 += [0] * (len(hist_2) - len(hist_1))
        for dist in bucket_dists:
            dists_day[dist.__name__] = dist(hist_1, hist_2)
        for dist in points_dists:
            dists_day[dist.__name__] = dist(points_1, points_2)
    except ValueError as e:
        print(metric, e)
        return
    return dists_day
