import pandas as pd
from toolz import compose

from ride_duration.utils import filter_ride_duration
from ride_duration.config import config


def preprocess(df: pd.DataFrame, train: bool):
    """Process and clean data for feature transformation."""

    df[config.CAT_FEATURES] = df[config.CAT_FEATURES].astype(str)
    df[config.NUM_FEATURES] = df[config.NUM_FEATURES].astype(float)

    if train:
        df = filter_ride_duration(df)
        return df[config.FEATURES + [config.TARGET]]
    else:
        return df[config.FEATURES]


def prepare_features(df: pd.DataFrame, transforms: tuple = (), train: bool = False):
    """Prepare data for model consumption."""

    df = preprocess(df, train=train)

    if train:
        y = df[config.TARGET].values if train else None
        X = df.drop([config.TARGET], axis=1)
    else:
        y = None
        X = df

    X = compose(*transforms[::-1])(X)
    return X, y
