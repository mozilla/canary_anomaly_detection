import pandas as pd

from canary.generator.utils import (
    _read_one_file,
    X_metric_to_df,
    y_metric_to_df,
    buckets_to_points,
    points_to_buckets,
    read_X_y_dicts_from_files,
)
from tests.generator.utils import assert_two_data_dicts_equal

DATA_DIFF_LEN = {
    'buckets': [0, 1, 3, 9, 26, 74, 212, 608, 1744, 5000],
    'kind': 'exponential',
    'data': {
        '20180312': [0.5, 0.2, 0.1, 0.05, 0.05, 0.03, 0.02, 0.02, 0.03],
        '20180313': [0.5, 0.2, 0.1, 0.05, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01],
    }
}

DATA_SAME_LEN = {
    'buckets': [0, 1, 3, 9, 26, 74, 212, 608, 1744, 5000],
    'kind': 'linear',
    'data': {
        '20180312': [0.5, 0.2, 0.1, 0.05, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01],
        '20180313': [0.5, 0.2, 0.1, 0.05, 0.03, 0.03, 0.05, 0.02, 0.02, 0.00],
    }
}

Y = {
    '20180312': 1,
    '20180313': 0,
}


def test_read_one_file():
    read_data = _read_one_file('tests/data_for_tests.json')

    assert_two_data_dicts_equal(DATA_DIFF_LEN, read_data)


def test_X_dict_to_df_different_length():
    X_df_true = pd.DataFrame({
        '20180312': [0.5, 0.2, 0.1, 0.05, 0.05, 0.03, 0.02, 0.02, 0.03, 0.0],
        '20180313': [0.5, 0.2, 0.1, 0.05, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01],
    }).sort_index(ascending=False)

    X_df = X_metric_to_df(DATA_DIFF_LEN)

    assert X_df.equals(X_df_true)


def test_X_dict_to_df_same_length():
    X_df_true = pd.DataFrame({
        '20180312': [0.5, 0.2, 0.1, 0.05, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01],
        '20180313': [0.5, 0.2, 0.1, 0.05, 0.03, 0.03, 0.05, 0.02, 0.02, 0.00],
    }, index=[0, 1, 3, 9, 26, 74, 212, 608, 1744, 5000]).sort_index(ascending=False)

    X_df = X_metric_to_df(DATA_SAME_LEN)

    assert X_df.equals(X_df_true)


def test_y_dict_to_df():
    y_df_true = pd.DataFrame({
        '20180312': 1,
        '20180313': 0,
    }, index=[0]).transpose()

    y_df = y_metric_to_df(Y)

    assert y_df.equals(y_df_true)


def test_to_points_and_back():
    data = points_to_buckets(buckets_to_points(DATA_SAME_LEN, 100))

    assert_two_data_dicts_equal(DATA_SAME_LEN, data)


def test_read_X_y_dicts_from_files():
    X_dict_true = {
        'data_for_tests.json': {
            'buckets': [0, 1, 3, 9, 26, 74, 212, 608, 1744, 5000],
            'kind': 'exponential',
            'data': {
                '20180312': [0.5, 0.2, 0.1, 0.05, 0.05, 0.03, 0.02, 0.02, 0.03],
                '20180313': [0.5, 0.2, 0.1, 0.05, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01],
            }
        }
    }
    y_dict_true = {
        'data_for_tests.json': {
            '20180312': 0,
            '20180313': 0,
        }
    }

    X_dict, y_dict = read_X_y_dicts_from_files(['tests/data_for_tests.json'])

    assert_two_data_dicts_equal(X_dict_true, X_dict)
    assert_two_data_dicts_equal(y_dict_true, y_dict)
