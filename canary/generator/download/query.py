def build_query_string(dates, metric, channel, version, os=None, os_version=None,
                       application=None, architecture=None):
    query_base = 'https://aggregates.telemetry.mozilla.org/aggregates_by/build_id/channels/'
    date_str = '%2C'.join(dates)
    os = '&os=' + os if os else ''
    os_version = '&osVersion=' + os_version if os_version else ''
    application = '&application=' + application if application else ''
    architecture = '&architecture=' + architecture if architecture else ''
    metric = '&metric=' + metric
    q = query_base + channel + '/?version=' + version + '&dates=' + date_str + \
        metric + os + os_version + application + architecture
    return q

