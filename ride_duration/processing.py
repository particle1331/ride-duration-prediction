import pandas as pd
from toolz import compose

from ride_duration.utils import filter_ride_duration
from ride_duration.config import config


def preprocess(df: pd.DataFrame, train: bool = True):
    """Fix types. Drop irrelevant data. Add and filter target column if train."""

    df[config.CATEGORICAL] = df[config.CATEGORICAL].astype(str)
    df[config.NUMERICAL] = df[config.NUMERICAL].astype(float)
    df = df[config.FEATURES]
    f = filter_ride_duration if train else lambda x: x
    return f(df)


def prepare_features(df: pd.DataFrame, transforms: tuple = (), train: bool = True):
    """Prepare data for model consumption at inference."""
    
    if not transforms:
        SELECTED_FEATURES = config.CATEGORICAL + config.NUMERICAL
        transforms = (lambda df: df[SELECTED_FEATURES],)

    df = preprocess(df, train=train)
    if train:
        y = df[config.TARGET].values if train else None
        X = df.drop(["duration"], axis=1)
    else:
        y = None
        X = df

    X = compose(*transforms[::-1])(X)
    return X, y
