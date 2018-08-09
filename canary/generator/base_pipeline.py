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

    def transform(self, X_dict, y_dict):
        for trans in self.transformers:
            if X_dict['kind'] in self.kinds:
                X_dict, y_dict = trans.transform(X_dict, y_dict=y_dict)
        return X_dict, y_dict
