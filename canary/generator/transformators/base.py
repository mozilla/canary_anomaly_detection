# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from abc import ABC


class Transformer(ABC):
    def __init__(self, kinds):
        """
        Abstract class that checks if the kinds of histograms are correct.
        :param kinds: Kinds on which the transformer should be evaluated
        """
        self.kinds = kinds

    def transform(self, X_dict, y_dict):
        raise NotImplementedError

    def check_kind(self, X_dict):
        if X_dict['kind'] not in self.kinds:
            raise ValueError('Transformer used in wrong pipeline')
