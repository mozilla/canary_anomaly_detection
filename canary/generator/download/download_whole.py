from collections import defaultdict
import json
from json import JSONDecodeError
import sys
import os

import requests

from canary.generator.download.query import build_query_string

if __name__ == '__main__':
    dates_versions_response = requests.get(
        'https://aggregates.telemetry.mozilla.org/aggregates_by/build_id/channels/nightly/dates/').text
    dates_versions = json.loads(dates_versions_response)
    # Dict consisting of pairs version: list of dates
    dates_for_versions = defaultdict(list)
    for item in dates_versions:
        dates_for_versions[item['version']].append(item['date'])
    versions = sorted(list(dates_for_versions.keys()))

    for version in versions[-int(sys.argv[2]):]:
        # Download options for a version
        options_response = requests.get(
            'https://aggregates.telemetry.mozilla.org/filters/?channel=nightly&version=' + version).text
        options = json.loads(options_response)
        # Make directories for a version
        try:
            os.makedirs(os.path.join(sys.argv[1], 'nightly_' + version + '_whole/'))
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
                os.path.join(sys.argv[1],
                             'nightly_' + version + '_whole/', metric + '.json'), 'w'))
