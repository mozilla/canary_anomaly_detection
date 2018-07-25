import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import truncexpon


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
        print('bad buckets!!!')
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
    print(bucket_dict['kind'])
    if bucket_dict['kind'] == 'categorical':
        points_dict['buckets_str'] = buckets
        points_dict['buckets'] = list(range(len(buckets)))
        buckets = points_dict['buckets']
    if bucket_dict['kind'] in ['exponential', 'categorical', 'enumerated', 'count', 'linear', 'boolean']:
        for date, hist in bucket_dict['data'].items():
            points = []
            for i, dens in enumerate(hist):
                low = buckets[i]
                if i + 1 == len(buckets):
                    high = low + (buckets[i] - buckets[i - 1])
                else:
                    high = buckets[i + 1]
                points += truncexpon.rvs(high, size=int(round(dens*n_points, 0)), loc=low).tolist()
            points_dict['data'][date] = points
    else:
        print(bucket_dict['kind'])
    return points_dict


def rec_bucket_add(point, i, bucket_data, buckets):
    try:
        if i == len(buckets) - 1:
            bucket_data[-1] += 1
            return bucket_data, i
        elif point < buckets[i + 1]:
            bucket_data[i] += 1
            return bucket_data, i
        else:
            return rec_bucket_add(point, i + 1, bucket_data, buckets)
    except IndexError:
        print(i, len(buckets))


def to_buckets(point_dict):
    bucket_dict = defaultdict(list)
    buckets = point_dict['buckets']
    bucket_dict['buckets'] = buckets
    bucket_dict['kind'] = point_dict['kind']
    bucket_dict['data'] = {}
    for date, points in point_dict['data'].items():
        if point_dict['kind'] in ['exponential', 'categorical', 'enumerated', 'count', 'linear', 'boolean']:
            bucket_data = [0]*len(buckets)
            i = 0
            for point in sorted(points):
                bucket_data, i = rec_bucket_add(point, i, bucket_data, buckets)
            bucket_dict['data'][date] = np.array(bucket_data)/sum(bucket_data)
        else:
            print(bucket_dict['kind'])
            break
    return bucket_dict
