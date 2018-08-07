# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from canary.generator.base_pipeline import DataTransformPipeline
from canary.generator.transformators.transformers_linear import (
    AddGaussianNoiseTransformer,
    AddToSomeBucketSmoothTransformer,
    AddMoveTransformer,
    AddAnotherDistTransformer,
)

anomaly_linear_pipeline = DataTransformPipeline([
    AddMoveTransformer(),
    AddMoveTransformer(),
    AddMoveTransformer(),
    AddMoveTransformer(),
    AddAnotherDistTransformer(),
    AddAnotherDistTransformer(),
    AddAnotherDistTransformer(),
    AddAnotherDistTransformer(),
    AddToSomeBucketSmoothTransformer(),
    AddToSomeBucketSmoothTransformer(),
    AddToSomeBucketSmoothTransformer(),
    AddToSomeBucketSmoothTransformer(),
], kinds=['linear'])

not_anomaly_linear_pipeline = DataTransformPipeline([
    # AddTrendTransformer(),
    # AddWeekSeasonalityTransformer(),
    AddGaussianNoiseTransformer(),
], kinds=['linear'])
