from collections import defaultdict
import json
from json import JSONDecodeError
from ast import literal_eval
import sys
import os

import requests

from .query import query

if __name__ == '__main__':

    dates_versions = requests.get(
        'https://aggregates.telemetry.mozilla.org/aggregates_by/build_id/channels/nightly/dates/').text
    dates_versions = literal_eval(dates_versions)
    dates_for_each = defaultdict(list)
    for item in dates_versions:
        dates_for_each[item['version']].append(item['date'])

    for v in dates_for_each.keys()[-5:]:
        options = requests.get(
            'https://aggregates.telemetry.mozilla.org/filters/?channel=nightly&version=' + v).text
        options = literal_eval(options)
        for m in options['metric'][:]:
            q = query(dates_for_each[v], m, 'nightly', v)
            data = requests.get(q).text
            try:
                data = json.loads(data)
            except JSONDecodeError:
                print(m)
                continue
            json.dump(data, open(
                os.path.join(sys.argv[1], 'nightly_' + v + '_whole/' + m + '.json'), 'w'))
