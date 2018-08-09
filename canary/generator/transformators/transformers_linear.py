# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from abc import ABC
from copy import deepcopy

import numpy as np

from .base import Transformer


class TransformerLinear(Transformer, ABC):
    def __init__(self):
        """
        Abstract class for all linear transformations.
        All of categorical transformers work on X_dict in point version.
        """
        super(TransformerLinear, self).__init__(['linear'])


class AddMoveTransformer(TransformerLinear):
    def __init__(self, shift=None, change_date=None):
        """
        Moves whole distribution. If the values are not provided, adds random shift
        in random day.

        The change is treated as an anomaly.

        :param shift: Value indicating how much to shift the data
        :param change_date: Day to be changed in format '%Y%m%d'
        """
        super(AddMoveTransformer, self).__init__()
        self.is_shift = False if shift is None else True
        self.is_date = False if change_date is None else True
        self.shift = shift
        self.change_date = change_date

    def transform(self, point_dict, y_dict):
        self.check_kind(point_dict)
        if not self.is_date:
            self.change_date = np.random.choice(list(point_dict['data'].keys()))
        if not self.is_shift:
            # The limits were chosen to best imitate actual anomalies
            self.shift = np.random.uniform(-0.8, 0.8) * max(point_dict['buckets'])

        y_copy = deepcopy(y_dict)
        points = point_dict['data'][self.change_date]
        points = abs(np.array(points) + self.shift)
        point_dict['data'][self.change_date] = list(points)
        y_copy[self.change_date] = 1
        return point_dict, y_copy


class AddToSomeBucketSmoothTransformer(TransformerLinear):
    def __init__(self, ratio=None, mean=None, std=None, change_date=None):
        """
        Changes ratio of data in specified day to gamma distribution with specified
        mean and std. If the values are not provided, changes the random ratio of data
        to be from gamma distribution of random mean and std.

        The change is treated as an anomaly.

        :param ratio: Ratio of data points to be changed
        :param mean: Mean of the gamma distribution
        :param std: Standard deviation of the gamma distribution
        :param change_date: Day to be changed in format '%Y%m%d'
        """
        super(AddToSomeBucketSmoothTransformer, self).__init__()
        self.is_mean = False if mean is None else True
        self.is_std = False if std is None else True
        self.is_ratio = False if ratio is None else True
        self.is_date = False if change_date is None else True
        self.ratio = ratio
        self.mean = mean
        self.std = std
        self.change_date = change_date

    def transform(self, point_dict, y_dict):
        self.check_kind(point_dict)
        y_copy = deepcopy(y_dict)
        if not self.is_date:
            self.change_date = np.random.choice(list(point_dict['data'].keys()))
        points = point_dict['data'][self.change_date]
        if not self.is_mean:
            self.mean = np.random.uniform(low=0, high=1) * max(point_dict['buckets'])
        if not self.is_std:
            self.std = np.random.uniform(low=0, high=1) * self.mean
        if not self.is_ratio:
            # The limits were chosen to best imitate actual anomalies
            self.ratio = np.random.uniform(0.1, 0.4)

        shape = (self.mean / self.std) ** 2
        scale = self.mean / shape
        points = [np.random.gamma(shape, scale) if
                  np.random.binomial(1, self.ratio) else p for p in points]
        point_dict['data'][self.change_date] = list(points)
        y_copy[self.change_date] = 1
        return point_dict, y_copy


class AddAnotherDistTransformer(TransformerLinear):
    def __init__(self, mean=None, std=None, change_date=None):
        """
        Changes data in specified day to gamma distribution with specified
        mean and std. If the values are not provided, changes the data to be from gamma
        distribution of random mean and std.

        The change is treated as an anomaly.

        :param mean: Mean of the gamma distribution
        :param std: Standard deviation of the gamma distribution
        :param change_date: Day to be changed in format '%Y%m%d'
        """
        super(AddAnotherDistTransformer, self).__init__()
        self.is_mean = False if mean is None else True
        self.is_std = False if std is None else True
        self.is_date = False if change_date is None else True
        self.mean = mean
        self.std = std
        self.change_date = change_date

    def transform(self, point_dict, y_dict):
        self.check_kind(point_dict)
        y_copy = deepcopy(y_dict)
        if not self.is_date:
            self.change_date = np.random.choice(list(point_dict['data'].keys()))
        if not self.is_mean:
            self.mean = np.random.uniform(low=0, high=1) * max(point_dict['buckets'])
        if not self.is_std:
            # The limits were chosen to best imitate actual anomalies
            self.std = np.random.uniform(low=0.3, high=1) * self.mean
        shape = (self.mean / self.std) ** 2
        scale = self.mean / shape
        points = point_dict['data'][self.change_date]
        points = np.random.gamma(shape, scale, len(points))
        point_dict['data'][self.change_date] = list(points)
        y_copy[self.change_date] = 1
        return point_dict, y_copy


class AddGaussianNoiseTransformer(TransformerLinear):
    def __init__(self, std=None):
        """
        Adds gaussian noise of defined standatd deviation to the data.
        If the value is not provided, adds gausian noise with random std.

        The change is NOT treated as an anomaly.

        :param std: Standard deviation of the noise
        """
        super(AddGaussianNoiseTransformer, self).__init__()
        self.is_std = False if std is None else True
        self.std = std

    def transform(self, point_dict, y_dict):
        self.check_kind(point_dict)
        if not self.is_std:
            # The limits were chosen so the change won't be too significant
            self.std = np.random.uniform(0, 0.01) * max(point_dict['buckets'])
        for date, points in point_dict['data'].items():
            noise = np.random.normal(0, self.std, len(points))
            points = np.array(points) + noise
            point_dict['data'][date] = list(points)
        return point_dict, y_dict


class AddTrendTransformer(TransformerLinear):
    def __init__(self, alpha=None):
        """
        Adds trend to the data. If value is no provided, the random trend is added.

        The change is NOT treated as an anomaly.

        :param alpha: Trend to be added
        """
        super(AddTrendTransformer, self).__init__()
        self.is_alpha = False if alpha is None else True
        self.alpha = alpha

    def transform(self, point_dict, y_dict):
        self.check_kind(point_dict)
        if not self.is_alpha:
            # The limits were chosen so the change won't be too significant
            self.alpha = np.random.uniform(-0.001, 0.001) * max(point_dict['buckets'])
        for i, (date, points) in enumerate(sorted(point_dict['data'].items())):
            trend = i * self.alpha
            points = np.array(points) + trend
            point_dict['data'][date] = list(points)
        return point_dict, y_dict


class AddWeekSeasonalityTransformer(TransformerLinear):
    def __init__(self, alpha=None):
        """
        Adds random weakly seasonality. Alpha can control how bir the changes are.
        If the value is not provided, the random alpha is chosen.

        The change is NOT treated as an anomaly.

        :param alpha: Controls how big the seasonality changes are
        """
        super(AddWeekSeasonalityTransformer, self).__init__()
        self.is_alpha = False if alpha is None else True
        self.alpha = alpha

    def transform(self, point_dict, y_dict):
        self.check_kind(point_dict)
        if not self.is_alpha:
            # The limits were chosen so the change won't be too significant
            self.alpha = np.random.uniform(0, 0.01)
        week = []
        for i in range(7):
            week.append(np.random.uniform(-1, 1) * max(point_dict['buckets']))
        for i, (date, points) in enumerate(sorted(point_dict['data'].items())):
            week_season = (week[i % 7]) * self.alpha
            points = np.array(points) + week_season
            point_dict['data'][date] = list(points)
        return point_dict, y_dict
