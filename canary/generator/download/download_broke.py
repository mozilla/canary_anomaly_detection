from collections import defaultdict
from multiprocessing import Pool
from itertools import product
import json
from json import JSONDecodeError
from ast import literal_eval
import os
import sys

import requests
import numpy as np

from .query import query


def get_data(arglist):
    dates_for_each, m, ch, v, os, app, ar = arglist
    q = query(dates_for_each, m, ch, v, os, application=app, architecture=ar)
    data = requests.get(q).text
    try:
        data = json.loads(data)
    except JSONDecodeError:
        return
    data['channel'] = ch
    data['metric'] = m
    data['version'] = v
    data['os'] = os
    data['application'] = app
    data['architecture'] = ar
    return data


if __name__ == '__main__':
    pool = Pool(20)
    channels = requests.get(
        'https://aggregates.telemetry.mozilla.org/aggregates_by/build_id/channels/').text
    channels = literal_eval(channels)
    ch = 'nightly'
    dates_versions = requests.get(
        'https://aggregates.telemetry.mozilla.org/aggregates_by/build_id/channels/' + ch + '/dates/').text
    dates_versions = literal_eval(dates_versions)
    dates_for_each = defaultdict(list)
    for item in dates_versions:
        dates_for_each[item['version']].append(item['date'])
    last_versions = sorted(dates_for_each.keys())[-1:]
    new_versions = {v: dates_for_each[v] for v in last_versions}
    for v in new_versions.keys():
        print(v)
        options = requests.get(
            'https://aggregates.telemetry.mozilla.org/filters/?channel=' + ch + '&version=' + v).text
        options = literal_eval(options)
        options['os'] = np.unique([os.split(',')[0] for os in options['os']])
        for m in options['metric']:
            whole_data_for_metric = []
            ops = list(product([dates_for_each[v]], [m], [ch], [v], options['os'],
                               options['application'], options['architecture']))
            for data in pool.imap_unordered(get_data, ops):
                whole_data_for_metric.append(data)
            json.dump(whole_data_for_metric,
                      open(os.path.join(sys.argv[1], 'nightly_' + v + '/' + m + '.json'), 'w'))
