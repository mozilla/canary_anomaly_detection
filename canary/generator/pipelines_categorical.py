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
