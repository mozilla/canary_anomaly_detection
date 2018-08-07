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


def process_file(filename):
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
            values = (np.array(measure['histogram']))/float(hist_sum)
            if measure_date in series:
                series['data'][measure_date] += np.array(values)
            else:
                series['data'][measure_date] = np.array(values)
    return series


def dict_to_df(data_dict):
    buckets = data_dict['buckets']
    data = data_dict['data']
    try:
        df = pd.DataFrame(data, index=buckets)[sorted(list(data.keys()))]
        df = df.sort_index(ascending=False)
    except ValueError:
        length = max([len(d) for d in data.values()])
        df = pd.DataFrame(data, index=list(range(length)))[sorted(list(data.keys()))]
        df = df.sort_index(ascending=False)
    return df


def y_dict_to_df(y_dict):
    df = pd.DataFrame(y_dict, index=[0]).transpose()
    return df


def to_points(bucket_dict, n_points=10000):
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
                points += truncexpon.rvs(
                    high, size=max(int(round(dens*n_points, 0)), 0), loc=low).tolist()
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


# TODO: There is still some bug: ValueError: Shape of passed values is (251, 9), indices imply (251, 11)
def bucket_add(point, i, bucket_data, buckets):
    while True:
        if i == len(buckets) - 1:
            bucket_data[-1] += 1
            return bucket_data, i
        elif point < buckets[i + 1]:
            bucket_data[i] += 1
            return bucket_data, i
        else:
            i += 1


def to_buckets(point_dict):
    bucket_dict = defaultdict(list)
    buckets = point_dict['buckets']
    bucket_dict['buckets'] = buckets
    bucket_dict['kind'] = point_dict['kind']
    bucket_dict['data'] = {}
    for date, points in point_dict['data'].items():
            bucket_data = [0]*len(buckets)
            i = 0
            for point in sorted(points):
                bucket_data, i = bucket_add(point, i, bucket_data, buckets)
            bucket_dict['data'][date] = list(np.array(bucket_data)/sum(bucket_data))
    return bucket_dict


def plot(X_true, y_true, X_changed, y_changed, name):
    a4_dims = (15, 6)
    plt.subplots(figsize=a4_dims)
    try:
        df = dict_to_df(X_true)
        y_df = y_dict_to_df(y_true)
    except ValueError:
        return
    sns.heatmap(df, cmap="YlGnBu")
    sns.heatmap(df, mask=1-y_df[0], cmap='YlOrRd')
    plt.xlabel('Date')
    plt.ylabel('Bucket')
    plt.title('NOT CHANGED')
    plt.savefig(name + '_NOT_CHANGED.png')
    plt.close()

    a4_dims = (15, 6)
    plt.subplots(figsize=a4_dims)
    df = dict_to_df(X_changed)
    y_df = y_dict_to_df(y_changed)
    sns.heatmap(df, cmap="YlGnBu")
    sns.heatmap(df, mask=1-y_df[0], cmap='YlOrRd')
    plt.xlabel('Date')
    plt.ylabel('Bucket')
    plt.title('CHANGED')
    plt.savefig(name + '_CHANGED.png')
    plt.close()


def read_X_y_from_files(list_of_files):
    """
    Reads files from disc and makes the y
    :param list_of_files: list of files with data
    :return: two dicts:
     hists - dict of histograms, where metrics are the keys and the dict of (date: histogram)
     pairs is the value, eg:
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
     y - dict of indicators if the day is anomalous, where metrics are the keys and the
     dict of (date: boolean (if the day is anomalous)) pairs is the value, eg:
     {
         'METRIC': {
             'buckets': [1, 2, 4, 8],
             'kind': 'exponential'
             'data': {
                 '20180713': 0,
                 '20180714': 1,
             }
         }
     }
    """
    hists = {}
    for f in sorted(list_of_files):
        name = f.split('/')[-1]
        try:
            if name not in hists.keys():
                hists[name] = process_file(f)
            hists[name]['data'] = {**hists[name]['data'], **process_file(f)['data']}
        except KeyError:
            print(f)

    y = {}
    for metric, hist in hists.items():
        y[metric] = dict.fromkeys(hist['data'], 0)
    return hists, y
