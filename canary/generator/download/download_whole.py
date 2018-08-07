# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from collections import defaultdict
import json
from json import JSONDecodeError
import os
from argparse import ArgumentParser

import requests

from canary.generator.download.query import build_query_string

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(dest='dir',
                        help='Directory, where generated data should be saved')
    parser.add_argument('-n', '--nightly_version_number', dest='n_versions',
                        help='Number of versions of nightly to download the data from',
                        default=5)
    args = parser.parse_args()
    import ipdb; ipdb.set_trace()
    dates_versions_response = requests.get(
        'https://aggregates.telemetry.mozilla.org/aggregates_by/build_id/channels/nightly/dates/').text
    dates_versions = json.loads(dates_versions_response)
    # Dict consisting of pairs version: list of dates
    dates_for_versions = defaultdict(list)
    for item in dates_versions:
        dates_for_versions[item['version']].append(item['date'])
    versions = sorted(list(dates_for_versions.keys()))

    for version in versions[-int(args.n_versions):]:
        # Download options for a version
        options_response = requests.get(
            'https://aggregates.telemetry.mozilla.org/filters/?channel=nightly&version=' + version).text
        options = json.loads(options_response)
        # Make directories for a version
        try:
            os.makedirs(os.path.join(args.dir, 'nightly_' + version + '_whole/'))
        except OSError:
            pass

        for metric in options['metric']:
            # Download data for metric
            q = build_query_string(dates_for_versions[version], metric, 'nightly', version)
            data_response = requests.get(q).text
            data = None
            try:
                data = json.loads(data_response)
            except JSONDecodeError:
                print('bad metric: ', metric)
                continue

            # Store data on disc
            json.dump(data, open(
                os.path.join(args.dir,
                             'nightly_' + version + '_whole/', metric + '.json'), 'w'))
