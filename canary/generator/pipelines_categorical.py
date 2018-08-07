# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from canary.generator.base_pipeline import DataTransformPipeline
from canary.generator.transformators.transformers_categorical import (
    AddNoiseTransformer,
    AddToSomeBucketTransformer,
    AddAnotherDistTransformer,
)

anomaly_categorical_pipeline = DataTransformPipeline([
    AddAnotherDistTransformer(),
    AddAnotherDistTransformer(),
    AddAnotherDistTransformer(),
    AddAnotherDistTransformer(),
    AddToSomeBucketTransformer(),
    AddToSomeBucketTransformer(),
    AddToSomeBucketTransformer(),
    AddToSomeBucketTransformer(),
], kinds=['categorical', 'enumerated', 'boolean', 'flag'])

not_anomaly_categorical_pipeline = DataTransformPipeline([
    AddNoiseTransformer(),
], kinds=['categorical', 'enumerated', 'boolean', 'flag'])
