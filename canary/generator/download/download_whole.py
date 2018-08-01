from collections import defaultdict
import json
from json import JSONDecodeError
import sys
import os

import requests

from canary.generator.download.query import build_query_string

if __name__ == '__main__':
    dates_versions = requests.get(
        'https://aggregates.telemetry.mozilla.org/aggregates_by/build_id/channels/nightly/dates/').text
    dates_versions = json.loads(dates_versions)
    # Dict consisting of pairs version: list of dates
    dates_for_versions = defaultdict(list)
    for item in dates_versions:
        dates_for_versions[item['version']].append(item['date'])
    versions = sorted(list(dates_for_versions.keys()))
    for version in versions[-int(sys.argv[2]):]:
        options = requests.get(
            'https://aggregates.telemetry.mozilla.org/filters/?channel=nightly&version=' + version).text
        options = json.loads(options)
        for m in options['metric'][:]:
            q = build_query_string(dates_for_versions[version], m, 'nightly', version)
            data = requests.get(q).text
            try:
                data = json.loads(data)
            except JSONDecodeError:
                print('bad metric: ', m)
                continue
            try:
                os.makedirs(os.path.join(sys.argv[1], 'nightly_' + version + '_whole/'))
            except OSError:
                pass
            json.dump(data, open(
                os.path.join(sys.argv[1],
                             'nightly_' + version + '_whole/', m + '.json'), 'w'))
