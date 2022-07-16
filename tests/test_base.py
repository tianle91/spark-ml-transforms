import pytest
from SparkMLTransforms.base import IdentityFitTransformer
from pandas.testing import assert_frame_equal
import pandas as pd

from pyspark.sql import SparkSession


@pytest.fixture(scope='package')
def spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()


@pytest.mark.parametrize(
    ('input_df', 'expected_df'),
    [
        pytest.param(
            pd.DataFrame({'a': [1, 2, 3]}),
            pd.DataFrame({'a_IdentityFitTransformer': [1, 2, 3]}),
            id='integer'
        ),
    ]
)
def test_IdentityFitTransformer(spark: SparkSession, input_df: pd.DataFrame, expected_df: pd.DataFrame):
    output_df = (
        IdentityFitTransformer(input_columns=['a'])
        .transform(spark.createDataFrame(input_df))
        .toPandas()
    )
    assert_frame_equal(output_df, expected_df)
