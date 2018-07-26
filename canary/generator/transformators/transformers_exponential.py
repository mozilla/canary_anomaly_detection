from abc import ABC
from copy import deepcopy

import numpy as np

from .base import Transformer


class TransformerExponential(Transformer, ABC):
    def __init__(self):
        super(TransformerExponential, self).__init__(['exponential', 'count'])


class AddMoveTransformer(TransformerExponential):
    def __init__(self, shift=None, change_date=None):
        """anomaly"""
        super(AddMoveTransformer, self).__init__()
        self.is_shift = False if shift is None else True
        self.is_date = False if change_date is None else True
        self.shift = shift
        self.change_date = change_date

    def transform(self, point_dict, y):
        if not self.is_date:
            self.change_date = np.random.choice(list(point_dict['data'].keys()))
        if not self.is_shift:
            self.shift = np.random.uniform(-0.8, 0.8)

        y_copy = deepcopy(y)
        points = point_dict['data'][self.change_date]
        points = abs(np.array(points) * (1 + self.shift))
        point_dict['data'][self.change_date] = list(points)
        if y is not None:
            y_copy[self.change_date] = 1
            return point_dict, y_copy
        return point_dict


class AddToSomeBucketSmoothTransformer(TransformerExponential):
    def __init__(self, percentage=None, mean=None, std=None, change_date=None):
        """anomaly"""
        super(AddToSomeBucketSmoothTransformer, self).__init__()
        self.is_mean = False if mean is None else True
        self.is_std = False if std is None else True
        self.is_percentage = False if percentage is None else True
        self.is_date = False if change_date is None else True
        self.percentage = percentage
        self.mean = mean
        self.std = std
        self.change_date = change_date

    def transform(self, point_dict, y):
        y_copy = deepcopy(y)
        if not self.is_date:
            self.change_date = np.random.choice(list(point_dict['data'].keys()))
        points = point_dict['data'][self.change_date]
        if not self.is_mean:
            self.mean = np.random.uniform(low=0, high=1) * max(point_dict['buckets'])
        if not self.is_std:
            self.std = np.random.uniform(low=0, high=1) * self.mean
        if not self.is_percentage:
            self.percentage = np.random.uniform(0.1, 0.4)

        shape = (self.mean / self.std) ** 2
        scale = self.mean / shape
        points = [np.random.gamma(shape, scale) if
                  np.random.binomial(1, self.percentage) else p for p in points]
        point_dict['data'][self.change_date] = list(points)
        if y is not None:
            y_copy[self.change_date] = 1
            return point_dict, y_copy
        return point_dict


class AddAnotherDistTransformer(TransformerExponential):
    def __init__(self, mean=None, std=None, change_date=None):
        """anomaly"""
        super(AddAnotherDistTransformer, self).__init__()
        self.is_mean = False if mean is None else True
        self.is_std = False if std is None else True
        self.is_date = False if change_date is None else True
        self.mean = mean
        self.std = std
        self.change_date = change_date

    def transform(self, point_dict, y):
        y_copy = deepcopy(y)
        if not self.is_date:
            self.change_date = np.random.choice(list(point_dict['data'].keys()))
        if not self.is_mean:
            self.mean = np.random.uniform(low=0, high=1) * max(point_dict['buckets'])
        if not self.is_std:
            self.std = np.random.uniform(low=0.3, high=1) * self.mean
        shape = (self.mean / self.std) ** 2
        scale = self.mean / shape
        points = point_dict['data'][self.change_date]
        points = np.random.gamma(shape, scale, len(points))
        point_dict['data'][self.change_date] = list(points)
        if y is not None:
            y_copy[self.change_date] = 1
            return point_dict, y_copy
        return point_dict


class AddExpNoiseTransformer(TransformerExponential):
    def __init__(self, percentage=None):
        """not an anomaly"""
        super(AddExpNoiseTransformer, self).__init__()
        self.is_percentage = False if percentage is None else True
        self.percentage = percentage

    def transform(self, point_dict, y):
        if not self.is_percentage:
            self.percentage = np.random.uniform(0, 0.2)
        for date, points in point_dict['data'].items():
            points = [np.random.exponential(abs(p)) if
                      np.random.binomial(1, self.percentage) else p for p in points]
            point_dict['data'][date] = list(points)
        if y is not None:
            return point_dict, y
        return point_dict


class AddTrendTransformer(TransformerExponential):
    def __init__(self, alpha=None):
        """not an anomaly, works on whole data"""
        super(AddTrendTransformer, self).__init__()
        self.is_alpha = False if alpha is None else True
        self.alpha = alpha

    def transform(self, point_dict, y):
        if not self.is_alpha:
            self.alpha = np.random.uniform(-0.001, 0.001)
        for i, (date, points) in enumerate(sorted(point_dict['data'].items())):
            trend = i * self.alpha
            points = np.array(points) * (1 + trend)
            point_dict['data'][date] = list(points)
        if y is not None:
            return point_dict, y
        return point_dict


class AddWeekSeasonalityTransformer(TransformerExponential):
    def __init__(self, alpha=None):
        """not an anomaly, works on whole data"""
        super(AddWeekSeasonalityTransformer, self).__init__()
        self.is_alpha = False if alpha is None else True
        self.alpha = alpha

    def transform(self, point_dict, y):
        if not self.is_alpha:
            self.alpha = np.random.uniform(0, 0.1)
        week = []
        for i in range(7):
            week.append(np.random.uniform(-1, 1))
        for i, (date, points) in enumerate(sorted(point_dict['data'].items())):
            week_season = (week[i % 7]) * self.alpha
            points = np.array(points) * (1 + week_season)
            point_dict['data'][date] = list(points)
        if y is not None:
            return point_dict, y
        return point_dict
