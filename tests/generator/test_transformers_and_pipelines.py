from canary.generator.base_pipeline import DataTransformPipeline
from tests.generator.utils import assert_two_data_dicts_equal
from canary.generator.transformators.transformers_categorical import (
    AddToSomeBucketTransformer as AddToSomeCategorcal,
    AddAnotherDistTransformer as AddAnotherDistCategorical,
    AddNoiseTransformer as AddNoiseCategorical,
)
from canary.generator.transformators.transformers_linear import (
    AddAnotherDistTransformer as AddAnotherDistLinear,
    AddGaussianNoiseTransformer as AddGaussianNoiseLinear,
    AddTrendTransformer as AddTrendLinear,
    AddMoveTransformer as AddMoveLinear,
    AddToSomeBucketSmoothTransformer as AddToSomeBucketSmoothLinear,
    AddWeekSeasonalityTransformer as AddWeekSeasonalityLinear,
)
from canary.generator.transformators.transformers_exponential import (
    AddWeekSeasonalityTransformer as AddWeekSeasonalityExp,
    AddToSomeBucketSmoothTransformer as AddToSomeBucketSmoothExp,
    AddMoveTransformer as AddMoveExp,
    AddTrendTransformer as AddTrendExp,
    AddAnotherDistTransformer as AddAnotherDistExp,
    AddExpNoiseTransformer as AddExpNoiseExp,
)
DATA_LINEAR = {
    'buckets': [0, 1, 3, 9, 26, 74, 212, 608, 1744, 5000],
    'kind': 'linear',
    'data': {
        '20180312': [0.5, 0.2, 0.1, 0.05, 0.05, 0.03, 0.01, 0.03, 0.02, 0.01],
        '20180313': [0.5, 0.2, 0.1, 0.05, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01],
    }
}
DATA_EXP = {
    'buckets': [0, 1, 3, 9, 26, 74, 212, 608, 1744, 5000],
    'kind': 'exponential',
    'data': {
        '20180312': [0.5, 0.2, 0.1, 0.05, 0.05, 0.03, 0.01, 0.03, 0.02, 0.01],
        '20180313': [0.5, 0.2, 0.1, 0.05, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01],
    }
}


def test_wrong_kinds_categorical():
    transformers = [
        AddToSomeCategorcal(),
        AddAnotherDistCategorical(),
        AddNoiseCategorical(),
    ]
    for transformer in transformers:
        pipeline = DataTransformPipeline([
            transformer,
        ], kinds=['linear'])
        error = ''

        try:
            pipeline.transform(DATA_LINEAR, y=None)
        except ValueError as e:
            error = str(e)

        assert error == 'Transformer used in wrong pipeline'


def test_wrong_kinds_linear():
    transformers = [
        AddAnotherDistLinear(),
        AddGaussianNoiseLinear(),
        AddMoveLinear(),
        AddToSomeBucketSmoothLinear(),
        AddTrendLinear(),
        AddWeekSeasonalityLinear(),

    ]
    for transformer in transformers:
        pipeline = DataTransformPipeline([
            transformer,
        ], kinds=['exponential'])
        error = ''

        try:
            pipeline.transform(DATA_EXP, y=None)
        except ValueError as e:
            error = str(e)

        assert error == 'Transformer used in wrong pipeline'


def test_wrong_kinds_exp():
    transformers = [
        AddAnotherDistExp(),
        AddExpNoiseExp(),
        AddMoveExp(),
        AddToSomeBucketSmoothExp(),
        AddTrendExp(),
        AddWeekSeasonalityExp(),

    ]
    for transformer in transformers:
        pipeline = DataTransformPipeline([
            transformer,
        ], kinds=['linear'])
        error = ''
        print(transformer)
        try:
            pipeline.transform(DATA_LINEAR, y=None)
        except ValueError as e:
            error = str(e)

        assert error == 'Transformer used in wrong pipeline'


def test_transform_on_wrong_kind():
    transformers = [
        AddToSomeCategorcal(),
        AddAnotherDistCategorical(),
        AddNoiseCategorical(),
    ]
    for transformer in transformers:
        pipeline = DataTransformPipeline([
            transformer,
        ], kinds=['categorical'])

        transformed_data, y = pipeline.transform(DATA_LINEAR, y=[0, 0])

        assert_two_data_dicts_equal(DATA_LINEAR, transformed_data)
