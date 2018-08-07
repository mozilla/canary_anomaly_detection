# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


class DataTransformPipeline:
    def __init__(self, transformers, kinds):
        """
        Base pipeline class.

        :param transformers: List of transformers to be used. Transformers should have
        the same kinds as the pipeline.
        :param kinds: List of kinds of histograms the pipeline should be performed on.
        """
        self.transformers = transformers
        self.kinds = kinds

    def transform(self, X, y=None):
        for trans in self.transformers:
            if X['kind'] in self.kinds:
                X, y = trans.transform(X, y=y)
        return X, y
