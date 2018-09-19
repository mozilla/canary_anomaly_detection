# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from collections import defaultdict
from multiprocessing import Pool
from itertools import product
import json
from json import JSONDecodeError
from ast import literal_eval
import os
from argparse import ArgumentParser

import requests
import numpy as np

from canary.generator.download.query import build_query_string


def _get_data(arglist):
    """
    Downloads the data for a specified parameters
    :param arglist: list of arguments, consists of:
    dates_for_versions, metric, channel, version, os, application, architecture
    :return: data in json format, eg:
    {
        "buckets": [0, 1, 2, 3],
        "data": [
            {"date": "20180625", "count": 11, "sum": 0, "histogram": [11, 0, 0, 0]},
            {"date": "20180624", "count": 19, "sum": 0, "histogram": [38, 0, 0, 0]},
            ...
        ]
        "kind": "enumerated",
        "description": "Some description",
        "channel": "nightly",
        "metric": "WEBVR_USERS_VIEW_IN",
        "version": "62",
        "os": "Linux",
        "application": "Fennec",
        "architecture": "arm"
    }
    """
    dates_for_versions, m, ch, v, os, app, ar = arglist
    q = build_query_string(dates_for_versions, m, ch, v,
                           os, application=app, architecture=ar)
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
    # parse the arguments
    parser = ArgumentParser()
    parser.add_argument(dest='dir',
                        help='Directory, where generated data should be saved')
    parser.add_argument('-ch', '--channel', dest='ch', default='nightly',
                        help='Name of the channel to download the data from')
    parser.add_argument('-n', '--number_of_versions', dest='n_versions', type=int,
                        help='Number of versions of nightly to download the data from',
                        default=5)
    args = parser.parse_args()

    main_url = 'https://aggregates.telemetry.mozilla.org/'
    pool = Pool(20)
    dates_versions_response = requests.get(
        main_url + 'aggregates_by/build_id/channels/' + args.ch + '/dates/').text
    dates_versions_response = literal_eval(dates_versions_response)
    dates_for_versions = defaultdict(list)
    for item in dates_versions_response:
        dates_for_versions[item['version']].append(item['date'])
    last_n_versions = sorted(dates_for_versions.keys())[-args.n_versions:]
    versions_with_dates = {v: dates_for_versions[v] for v in last_n_versions}

    for v in versions_with_dates.keys():
        try:
            os.makedirs(os.path.join(args.dir, 'nightly_' + v))
        except OSError:
            pass
        options = requests.get(
            main_url + 'filters/?channel=' + args.ch + '&version=' + v).text
        options = literal_eval(options)
        options['os'] = np.unique([os.split(',')[0] for os in options['os']])
        for m in options['metric']:
            whole_data_for_metric = []
            ops = list(product(
                [dates_for_versions[v]], [m], [args.ch], [v], options['os'],
                options['application'], options['architecture']
            ))
            for data in pool.imap_unordered(_get_data, ops):
                whole_data_for_metric.append(data)
            # the data is saved as a list of elements from `_get_data`, see docstring
            with open(os.path.join(args.dir,
                                   'nightly_' + v + '/' + m + '.json'), 'w') as file:
                json.dump(whole_data_for_metric, file)
