from typing import Dict, List, Optional

import xgboost as xgb


def get_splits(data, label, params: Optional[dict] = None) -> Dict[str, List[float]]:
    """Return features and their splits resulting from a xgboost fit.

    Args:
        data: data argument to xgb.DMatrix
        label: label argument to xgb.DMatrix
        params (Optional[dict], optional): params pargment to xgb.train. Defaults to None.
    """
    params = {} if params is None else params
    dtrain = xgb.DMatrix(data=data, label=label)
    booster = xgb.train(params=params, dtrain=dtrain)
    trees_df = booster.trees_to_dataframe()
    trees_df = trees_df.loc[trees_df['Split'].notna(), :]
    return {
        feature: sorted(list(subdf['Split']))
        for feature, subdf in trees_df.groupby('Feature')
    }
