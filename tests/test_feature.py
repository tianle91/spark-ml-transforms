from sklearn.datasets import fetch_california_housing

from SparkMLTransforms.feature import get_splits


def test_get_splits():
    data = fetch_california_housing(as_frame=True)
    splits = get_splits(data=data['data'], label=data['target'])
    assert set(splits.keys()) == set(data['feature_names'])
    for vals in splits.values():
        assert all([isinstance(v, float) for v in vals]), 'Not all vals are floats'
