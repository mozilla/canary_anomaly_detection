# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from abc import ABC


class Transformer(ABC):
    def __init__(self, kinds):
        self.kinds = kinds

    def transform(self, X, y):
        if X['kind'] in self.kinds:
            raise ValueError
        raise NotImplementedError
