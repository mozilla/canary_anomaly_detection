# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import truncexpon
from matplotlib import pyplot as plt
import seaborn as sns


def _read_one_file(filename):
    """
    Reads one file and parse it to the format provided in docs for
    `read_X_y_dicts_from_files`. Changes histogram values from count to density.
    :param filename: Name of the file to be read
    :return: The dict containing daa for one metric, eg:
    {
        'buckets': [1, 2, 4, 8],
        'kind': 'exponential'
        'data': {
            '20180713': [0.4, 0.2, 0.1, 0.3],
            '20180714': [0.5, 0.3, 0.0, 0.2],
        }
    }
    """
    series = dict()
    series['data'] = {}
    with open(filename) as f:
        data = json.load(f)
        series['buckets'] = data['buckets']
        series['kind'] = data['kind']
        measures = data['data']
        for measure in measures:
            measure_date = measure['date']
            hist_sum = sum(measure['histogram'])
            values = (np.array(measure['histogram'])) / float(hist_sum)
            if measure_date in series:
                series['data'][measure_date] += np.array(values)
            else:
                series['data'][measure_date] = np.array(values)
    return series


def X_metric_to_df(X_metric):
    """
    Parses one metric from X_dict do data frame. The metric should look like:
    {
        'buckets': [1, 2, 4, 8],
        'kind': 'exponential'
        'data': {
            '20180713': [0.4, 0.2, 0.1, 0.3],
            '20180714': [0.5, 0.3, 0.0, 0.2],
        }
    }
    :param X_metric: One metric from X_dict to be parsed
    :return: Data frame with days as columns and buckets as rows
    """
    buckets = X_metric['buckets']
    data = X_metric['data']
    try:
        X_df = pd.DataFrame(data, index=buckets)[sorted(list(data.keys()))]
        X_df = X_df.sort_index(ascending=False)
    except ValueError:
        length = max([len(hist) for hist in data.values()])
        data = {day: np.pad(hist, (0, length - len(hist)), 'constant', constant_values=0)
                for day, hist in data.items()}
        X_df = pd.DataFrame(data, index=list(range(length)))[sorted(list(data.keys()))]
        X_df = X_df.sort_index(ascending=False)
    return X_df


def y_metric_to_df(y_metric):
    """
    Parses one metric from y_dict do data frame. The metric should look like:
    {
        '20180713': 0,
        '20180714': 1,
    }
    :param y_metric: One metric from y_dict to be parsed
    :return: Data frame with days as columns and one row indicating if
    the day was anomalous
    """
    y_df = pd.DataFrame(y_metric, index=[0]).transpose()
    return y_df


def buckets_to_points(bucket_dict, n_points=10000):
    """
    Transforms bucket_dict to point_dict by generating n_points from the distribution,
    that depends on the kind of histogram. The points correspond to the histogram, so:
    `bucket_dict` is the same as `points_to_buckets(buckets_to_points(bucket_data))`

    Transforming to points is necessary for some transformations.

    :param bucket_dict: one metric from X_dict in bucket version to be transformed
    :param n_points: Number of points to be generated. Greater number of points indicates
    more accurate point distribution and more time that is needed.
    :return: one metric from X_dict in the point version
    """
    points_dict = defaultdict(list)
    buckets = bucket_dict['buckets']
    points_dict['buckets'] = buckets
    points_dict['kind'] = bucket_dict['kind']
    points_dict['data'] = {}
    if bucket_dict['kind'] == 'categorical':
        points_dict['buckets_str'] = buckets
        points_dict['buckets'] = list(range(len(buckets)))
        buckets = points_dict['buckets']
    if bucket_dict['kind'] in ['exponential', 'count']:
        for date, hist in bucket_dict['data'].items():
            points = []
            for i, dens in enumerate(hist):
                low = buckets[i]
                if i + 1 == len(buckets):
                    high = low + (buckets[i] - buckets[i - 1])
                else:
                    high = buckets[i + 1]
                points += truncexpon.rvs(high,
                                         size=max(int(round(dens*n_points, 0)), 0),
                                         loc=low).tolist()
            points_dict['data'][date] = list(points)
    else:
        for date, hist in bucket_dict['data'].items():
            points = []
            if len(hist) != len(buckets):
                buckets = list(range(len(hist)))
                points_dict['buckets'] = buckets
            for i, dens in enumerate(hist):
                low = buckets[i]
                if i + 1 == len(buckets):
                    high = low + (buckets[i] - buckets[i - 1])
                else:
                    high = buckets[i + 1]
                points += np.random.uniform(high=high, low=low,
                                            size=int(round(dens*n_points, 0))).tolist()
            points_dict['data'][date] = list(points)
    return points_dict


def _bucket_add(point, i, bucket_data, buckets):
    """
    Adds the point to corresponding bucket
    :param point: (int) Data point
    :param i: Current index
    :param bucket_data: Data in bucket version (one metric from X_dict)
    :param buckets: Buckets list for
    :return: bucket_data with a point added and new index
    """
    while True:
        if i == len(buckets) - 1:
            bucket_data[-1] += 1
            return bucket_data, i
        elif point < buckets[i + 1]:
            bucket_data[i] += 1
            return bucket_data, i
        else:
            i += 1


def points_to_buckets(point_dict):
    """
    Transforms the data in point version again to bucket version. The points correspond
    to the histogram, so:`bucket_dict` is the same as
    `points_to_buckets(buckets_to_points(bucket_data))`
    :param point_dict: one metric from X_dict in point version to be transformed
    :return: one metric from X_dict in the bucket version
    """
    bucket_dict = defaultdict(list)
    buckets = point_dict['buckets']
    bucket_dict['buckets'] = buckets
    bucket_dict['kind'] = point_dict['kind']
    bucket_dict['data'] = {}
    for date, points in point_dict['data'].items():
            bucket_data = [0]*len(buckets)
            i = 0
            for point in sorted(points):
                bucket_data, i = _bucket_add(point, i, bucket_data, buckets)
            bucket_dict['data'][date] = list(np.array(bucket_data)/sum(bucket_data))
    return bucket_dict


def plot(X_true, y_true, X_changed, y_changed, filename):
    """
    Plots the changes in data. Produces two plots: one of untouched data and one of data
    with anomalies. The anomalies are coloured with red.
    :param X_true: One unchanged metric from X_dict in bucket version
    :param y_true: One unchanged metric from y_dict
    :param X_changed: One metric from X_dict with anomalies in bucket version
    :param y_changed: One metric from y_dict with anomalies
    :param filename: Directory, where the plot should be saved
    """
    # It's used to have the same colour range on both plots
    vmax = np.max([np.max(x) for x in X_true['data'].values()] +
                  [np.max(x) for x in X_changed['data'].values()])
    vmin = 0

    plt.figure(1, figsize=(15, 15))
    plt.subplot(211)
    try:
        df = X_metric_to_df(X_true)
        y_df = y_metric_to_df(y_true)
    except ValueError:
        return
    sns.heatmap(df, cmap="YlGnBu", vmax=vmax, vmin=vmin)
    sns.heatmap(df, mask=1 - y_df[0], cmap='YlOrRd', vmax=vmax, vmin=vmin)
    plt.xlabel('Date')
    plt.ylabel('Bucket')
    plt.title('NOT CHANGED')

    plt.subplot(212)
    df = X_metric_to_df(X_changed)
    y_df = y_metric_to_df(y_changed)
    sns.heatmap(df, cmap="YlGnBu", vmax=vmax, vmin=vmin)
    sns.heatmap(df, mask=1 - y_df[0], cmap='YlOrRd', vmax=vmax, vmin=vmin)
    plt.xlabel('Date')
    plt.ylabel('Bucket')
    plt.title('CHANGED')
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(filename + '_PLOT.png')
    plt.close()


def read_X_y_dicts_from_files(list_of_files):
    """
    Reads files from disc and makes the y_dict
    :param list_of_files: list of files with data
    :return: two dicts:
     X_dict - dict of histograms, where metrics are the keys and the dict of
     (date: histogram) pairs is the value, eg:
     {
         'METRIC': {
             'buckets': [1, 2, 4, 8],
             'kind': 'exponential'
             'data': {
                 '20180713': [0.4, 0.2, 0.1, 0.3],
                 '20180714': [0.5, 0.3, 0.0, 0.2],
             }
         }
     }
     y_dict - dict of indicators if the day is anomalous, where metrics are the keys and
     the dict of (date: boolean (if the day is anomalous)) pairs is the value, eg:
     {
         'METRIC': {
             '20180713': 0,
             '20180714': 1,
         }
     }
    """
    X_dict = {}
    for f in sorted(list_of_files):
        name = f.split('/')[-1]
        try:
            if name not in X_dict.keys():
                X_dict[name] = _read_one_file(f)
            X_dict[name]['data'] = {**X_dict[name]['data'], **_read_one_file(f)['data']}
        except KeyError:
            print(f)

    y_dict = {}
    for metric, hist in X_dict.items():
        y_dict[metric] = dict.fromkeys(hist['data'], 0)
    return X_dict, y_dict
