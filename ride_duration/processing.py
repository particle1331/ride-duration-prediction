import pandas as pd

from ride_duration.utils import create_target_column, filter_ride_duration
from ride_duration.config import config


def preprocess(data: pd.DataFrame, target: bool, filter_target: bool = False):
    """Process and clean data for feature transformation."""

    data[config.CAT_FEATURES] = data[config.CAT_FEATURES].astype(str)
    data[config.NUM_FEATURES] = data[config.NUM_FEATURES].astype(float)

    if target:
        data = create_target_column(data)
        if filter_target:
            data = filter_ride_duration(data)

        X = data[config.FEATURES]
        y = data[config.TARGET].values

        return X, y

    else:
        X = data[config.FEATURES]
        return X
