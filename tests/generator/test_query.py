from canary.generator.download.query import build_query_string


def test_query():
    query_true = 'https://aggregates.telemetry.mozilla.org/aggregates_by/build_id/' \
                 'channels/nightly/?version=61&dates=20180712%2C20180713&metric=METRIC&' \
                 'application=Fennec&architecture=x86'

    query_gen = build_query_string(['20180712', '20180713'], 'METRIC', 'nightly',
                                   '61', application='Fennec', architecture='x86')

    assert query_gen == query_true
