from abc import ABC
from copy import deepcopy

import numpy as np

from .base import Transformer


class TransformerCategorical(Transformer, ABC):
    def __init__(self):
        super(TransformerCategorical, self).__init__(['categorical', 'enumerated',
                                                      'boolean', 'flag'])


class AddToSomeBucketTransformer(TransformerCategorical):
    def __init__(self, percentage=None, bucket_number=None, change_date=None):
        """anomaly, works on buckets!!!"""
        super(AddToSomeBucketTransformer, self).__init__()
        self.is_bucket = False if bucket_number is None else True
        self.is_percentage = False if percentage is None else True
        self.is_date = False if change_date is None else True
        self.percentage = percentage
        self.bucket_number = bucket_number
        self.change_date = change_date

    def transform(self, bucket_dict, y):
        bucket_dict_copy = deepcopy(bucket_dict)
        y_copy = deepcopy(y)
        if not self.is_date:
            self.change_date = np.random.choice(list(bucket_dict_copy['data'].keys()))
        hist = bucket_dict_copy['data'][self.change_date]
        if not self.is_bucket:
            self.bucket_number = np.random.choice(
                list(range(len(hist))))
        if not self.percentage:
            self.percentage = np.random.uniform(0.2, 0.6)

        hist[self.bucket_number] += self.percentage
        hist = hist / (sum(hist))
        bucket_dict_copy['data'][self.change_date] = hist
        if y is not None:
            y_copy[self.change_date] = 1
            return bucket_dict_copy, y_copy
        return bucket_dict_copy


class AddAnotherDistTransformer(TransformerCategorical):
    def __init__(self, mean=None, std=None, change_date=None):
        """anomaly, works on buckets!!!"""
        super(AddAnotherDistTransformer, self).__init__()
        self.is_mean = False if mean is None else True
        self.is_std = False if std is None else True
        self.is_date = False if change_date is None else True
        self.mean = mean
        self.std = std
        self.change_date = change_date

    def transform(self, bucket_dict, y):
        y_copy = deepcopy(y)
        if not self.is_date:
            self.change_date = np.random.choice(list(bucket_dict['data'].keys()))
        if not self.is_mean:
            self.mean = np.random.uniform(low=0, high=1) * len(bucket_dict['buckets'])
        if not self.is_std:
            self.std = np.random.uniform(low=0.3, high=1) * self.mean
        shape = (self.mean / self.std) ** 2
        scale = self.mean / shape
        hist = bucket_dict['data'][self.change_date]
        hist = np.random.gamma(shape, scale, len(hist))
        bucket_dict['data'][self.change_date] = hist/sum(hist)
        if y is not None:
            y_copy[self.change_date] = 1
            return bucket_dict, y_copy
        return bucket_dict


class AddNoiseTransformer(TransformerCategorical):
    def __init__(self, std=None):
        """not an anomaly"""
        super(AddNoiseTransformer, self).__init__()
        self.is_std = False if std is None else True
        self.std = std

    def transform(self, bucket_dict, y):
        if not self.is_std:
            self.std = np.random.uniform(0, 0.001) * len(bucket_dict['buckets'])
        for date, hist in bucket_dict['data'].items():
            noise = np.random.normal(0, self.std, len(hist))
            hist += noise
            hist = abs(hist)/sum(abs(hist))
            bucket_dict['data'][date] = hist
        if y is not None:
            return bucket_dict, y
        return bucket_dict
