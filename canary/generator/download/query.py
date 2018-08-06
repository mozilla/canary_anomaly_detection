def build_query_string(dates, metric, channel, version, os=None, os_version=None,
                       application=None, architecture=None):
    query_base = 'https://aggregates.telemetry.mozilla.org/aggregates_by/build_id/channels/'
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
