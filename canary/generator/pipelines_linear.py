from canary.generator.base_pipeline import DataTransformPipeline
from canary.generator.transformators.transformers_linear import (
    AddWeekSeasonalityTransformer,
    AddGaussianNoiseTransformer,
    AddToSomeBucketSmoothTransformer,
    AddMoveTransformer,
    AddAnotherDistTransformer,
    AddTrendTransformer
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
