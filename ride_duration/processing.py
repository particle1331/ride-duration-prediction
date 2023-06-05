import pandas as pd

from ride_duration.utils import filter_ride_duration
from ride_duration.config import config


def preprocess(df: pd.DataFrame, train: bool):
    """Process and clean data for feature transformation."""

    df[config.CAT_FEATURES] = df[config.CAT_FEATURES].astype(str)
    df[config.NUM_FEATURES] = df[config.NUM_FEATURES].astype(float)

    if train:
        df = filter_ride_duration(df)
        X = df[config.FEATURES]
        y = df[config.TARGET].values
        return X, y
    else:
        X = df[config.FEATURES]
        return X
