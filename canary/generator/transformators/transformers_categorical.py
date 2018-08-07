# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from abc import ABC
from copy import deepcopy

import numpy as np

from .base import Transformer


class TransformerCategorical(Transformer, ABC):
    def __init__(self):
        """
        Abstract class for all categorical transformations.
        All of categorical transformers work on X_dict in bucket version.
        """
        super(TransformerCategorical, self).__init__(
            ['categorical', 'enumerated', 'boolean', 'flag']
        )


class AddToSomeBucketTransformer(TransformerCategorical):
    def __init__(self, val_to_add=None, bucket_number=None, change_date=None):
        """
        Adds val_to_add to the bucket in specified day. If the values are not provided,
        adds random value to random bucket in random day.
        Normalizes histogram again to density after adding value.

        The change is treated as an anomaly.

        :param val_to_add: Value to be added
        :param bucket_number: Number of the bucket
        :param change_date: Day to be changed in format '%Y%m%d'
        """
        super(AddToSomeBucketTransformer, self).__init__()
        self.is_bucket = False if bucket_number is None else True
        self.is_val_to_add = False if val_to_add is None else True
        self.is_date = False if change_date is None else True
        self.val_to_add = val_to_add
        self.bucket_number = bucket_number
        self.change_date = change_date

    def transform(self, bucket_dict, y):
        self.check_kind(bucket_dict)
        bucket_dict_copy = deepcopy(bucket_dict)
        y_copy = deepcopy(y)
        if not self.is_date:
            self.change_date = np.random.choice(list(bucket_dict_copy['data'].keys()))
        hist = bucket_dict_copy['data'][self.change_date]
        if not self.is_bucket:
            self.bucket_number = np.random.choice(
                list(range(len(hist))))
        if not self.val_to_add:
            self.val_to_add = np.random.uniform(0.2, 0.6)

        hist[self.bucket_number] += self.val_to_add
        hist = hist / (sum(hist))
        bucket_dict_copy['data'][self.change_date] = list(hist)
        if y is not None:
            y_copy[self.change_date] = 1
            return bucket_dict_copy, y_copy
        return bucket_dict_copy


class AddAnotherDistTransformer(TransformerCategorical):
    def __init__(self, mean=None, std=None, change_date=None):
        """
        Replaces a whole day with entirely different gamma distribution. If the values are
        not provided, the random mean, std and change_date are chosen.

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

    def transform(self, bucket_dict, y):
        self.check_kind(bucket_dict)
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
        bucket_dict['data'][self.change_date] = list(hist/sum(hist))
        if y is not None:
            y_copy[self.change_date] = 1
            return bucket_dict, y_copy
        return bucket_dict


class AddNoiseTransformer(TransformerCategorical):
    def __init__(self, std=None):
        """
        Adds gaussian noise to all days in the data. If std is not provided standard
        deviation is chosen randomly.

        The change is NOT treated as an anomaly.

        :param std: Standard deviation of the gaussian noise
        """
        super(AddNoiseTransformer, self).__init__()
        self.is_std = False if std is None else True
        self.std = std

    def transform(self, bucket_dict, y):
        self.check_kind(bucket_dict)
        if not self.is_std:
            self.std = np.random.uniform(0, 0.001) * len(bucket_dict['buckets'])
        for date, hist in bucket_dict['data'].items():
            noise = np.random.normal(0, self.std, len(hist))
            hist += noise
            hist = abs(hist)/sum(abs(hist))
            bucket_dict['data'][date] = list(hist)
        if y is not None:
            return bucket_dict, y
        return bucket_dict
