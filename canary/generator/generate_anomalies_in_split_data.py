# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import json
from collections import defaultdict
import os
import pickle
from copy import deepcopy
from argparse import ArgumentParser

import numpy as np


def aggregate(data, buckets, field):
    """
    Aggregates the data by the field
    :param data: split data for one metric
    :param buckets: buckets for one metric
    :param field: field by which to aggregate
    :return: aggregated data
    """
    agg_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: np.zeros(len(buckets))))
    )
    for opts in data:
        for date, hist in opts['data'].items():
            try:
                agg_data[opts[field]]['data'][date] += hist
            except ValueError:
                old = agg_data[opts[field]]['data'][date]
                if len(old) < len(hist):
                    agg_data[opts[field]]['data'][date] = np.concatenate(
                        [old, np.zeros(len(hist) - len(old))]
                    )
                elif len(old) > len(hist):
                    hist = np.concatenate([hist, np.zeros(len(old) - len(hist))])
                agg_data[opts[field]]['data'][date] += hist

    for arch, data in agg_data.items():
        agg_data[arch]['buckets'] = buckets
        agg_data[arch]['data'] = {date: hist/np.sum(hist)
                                  for date, hist in data['data'].items()}
    return agg_data


def defaultdict_to_dict(def_dict):
    """
    Changes nested defaultdict into nested dict. It's necessary for example for pickling.
    :param def_dict: defaultdict to be changed
    :return: nested dict with the same structure as input defaultdict
    """
    if isinstance(def_dict, (defaultdict, dict)):
        normal_dict = dict(def_dict)
        for k, v in normal_dict.items():
            normal_dict[k] = defaultdict_to_dict(v)
        return normal_dict
    else:
        return def_dict


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(dest='data_dir',
                        help='Directory with previously generated data', nargs='*')
    parser.add_argument('-s', '--split_data_dir', dest='split_data_dir',
                        help='Directory with downloaded split data', nargs='*')
    parser.add_argument('-o', '--output_dir', dest='output_dir',
                        help='Directory where the transformed data should be saved')
    args = parser.parse_args()
    X_test = dict()
    for file in [path for path in args.data_dir if path.endswith('_X_test.json')]:
        m = file.split('/')[-1].split('_X_test')[0]
        with open(file) as f:
            X_test[m] = json.load(f)

    y_true = dict()
    for file in [path for path in args.data_dir if path.endswith('_y_test.json')]:
        m = file.split('/')[-1].split('_y_test')[0]
        with open(file) as f:
            y_true[m] = json.load(f)

    data_arch = dict()
    data_oses = dict()
    whole_data = dict()
    for directory in [path for path in args.data_dir if path.endswith('_X_test.json')]:
        m = directory.split('data/')[-1].split('_X_test')[0]
        split_data = []
        try:
            for split_dir in [path for path in args.split_data_dir
                               if path.endswith('/' + m + '.json')]:
                with open(split_dir) as file:
                    split_data += json.load(file)
            with open(directory) as file:
                whole_data[m] = json.load(file)
            test_dates = set(whole_data[m]['data'].keys())

            # filtering
            split_data = [data for data in split_data if data is not None]
            split_data = [data for data in split_data if
                           data['application'] == 'Firefox']
            for opts in split_data:
                opts['data'] = {data['date']: np.array(data['histogram'])
                                for data in opts['data']}

            # architectures
            data_arch[m] = aggregate(split_data, whole_data[m]['buckets'],
                                     'architecture')

            # oses
            data_oses[m] = aggregate(split_data, whole_data[m]['buckets'], 'os')

        except (KeyError, FileNotFoundError) as e:
            print(m)
            print(e)

    data_arch_changed = deepcopy(data_arch)
    data_oses_changed = deepcopy(data_oses)
    for m in X_test.keys():
        for day, anom in y_true[m].items():
            whole_hist = whole_data[m]['data'][day]
            if anom == 1:
                categories = []
                for arch in data_arch[m].keys():
                    if data_arch[m][arch]['data'].get(day) is not None:
                        categories.append(arch)
                for os_name in data_oses[m].keys():
                    if data_oses[m][os_name]['data'].get(day) is not None:
                        categories.append(os_name)
                if len(categories) == 0:
                    continue
                number = np.random.randint(1, len(categories) + 1)
                changed_cats = np.random.choice(categories, number, replace=False)
                for cat in changed_cats:
                    change = np.random.random(1)[0]
                    if cat in data_oses[m].keys():
                        data_oses_changed[m][cat]['data'][day] = \
                            change * np.array(whole_hist) + \
                            (1 - change) * np.array(data_oses[m][cat]['data'][day])

                    else:
                        data_arch_changed[m][cat]['data'][day] = \
                            change * np.array(whole_hist) + \
                            (1 - change) * np.array(data_arch[m][cat]['data'][day])

    with open(os.path.join(args.output_dir, 'data_split_architecture'), 'wb') as file:
        pickle.dump(defaultdict_to_dict(data_arch), file)
    with open(os.path.join(args.output_dir, 'data_split_architecture_CHANGED'), 'wb') as file:
        pickle.dump(defaultdict_to_dict(data_arch_changed), file)
    with open(os.path.join(args.output_dir, 'data_split_os'), 'wb') as file:
        pickle.dump(defaultdict_to_dict(data_oses), file)
    with open(os.path.join(args.output_dir, 'data_split_os_CHANGED'), 'wb') as file:
        pickle.dump(defaultdict_to_dict(data_oses_changed), file)
