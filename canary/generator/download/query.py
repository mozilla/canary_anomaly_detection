# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


def build_query_string(dates, metric, channel, version, os=None, os_version=None,
                       application=None, architecture=None):
    """
    Builds query string for Telemetry HTTP API. For possible strings for each variable
    check Telemetry HTTP API documentation.
    :param dates: List of dates in format '%Y%m%d', eg: ['20180723', '20180724']
    :param metric: Metric, eg: ''
    :param channel: Channel, eg: 'nightly'
    :param version: Version of the browser, eg: '61'
    :param os: OS, eg: 'Windows_NT'
    :param os_version: Version of the OS, eg: '6.1'
    :param application: Exact application, eg: 'Fennec'
    :param architecture: Processor architecture, eg: 'x86'
    :return: query string
    """
    query_base = 'https://aggregates.telemetry.mozilla.org/' \
                 'aggregates_by/build_id/channels/'
    date_str = '%2C'.join(dates)
    query = query_base + channel + '/?version=' + version + \
        '&dates=' + date_str + '&metric=' + metric

    if os:
        query += '&os=' + os
    if os_version:
        query += '&osVersion=' + os_version
    if application:
        query += '&application=' + application
    if architecture:
        query += '&architecture=' + architecture
    return query
