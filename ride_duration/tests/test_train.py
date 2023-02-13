import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.errors import SettingWithCopyWarning
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import DictVectorizer

from ride_duration.utils import preprocess, convert_to_dict, plot_duration_histograms
from ride_duration.config import DATASET_DIR, config

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def prepare_dataset(filename: str):
    # Load and preprocess data
    df = pd.read_parquet(DATASET_DIR / filename)
    df = preprocess(df)

    # Feature engineering
    pass

    # Feature selection
    SELECTED_FEATURES = config.CATEGORICAL + config.NUMERICAL
    y = df[config.TARGET].values
    X = df[SELECTED_FEATURES]

    return X, y


def test_preprocess():
    filename = "green_tripdata_2021-01.parquet"
    df = pd.read_parquet(DATASET_DIR / filename)
    orig_shape = df.shape
    df = preprocess(df)

    assert sorted(list(df.columns)) == sorted(config.FEATURES + [config.TARGET])
    assert df.shape[0] <= orig_shape[0]


def test_pipeline():
    train = "green_tripdata_2021-01.parquet"
    valid = "green_tripdata_2021-02.parquet"

    # Get modeling dataset
    X_train, y_train = prepare_dataset(train)
    X_valid, y_valid = prepare_dataset(valid)

    # Fit model pipeline
    pipe = make_pipeline(
        FunctionTransformer(convert_to_dict), 
        DictVectorizer(), 
        LinearRegression()
    )
    pipe.fit(X_train, y_train)

    # Check performance
    p_train = pipe.predict(X_train)
    p_valid = pipe.predict(X_valid)
    mse_train = mean_squared_error(y_train, p_train, squared=False)
    mse_valid = mean_squared_error(y_valid, p_valid, squared=False)

    assert mse_train < 10
    assert mse_valid < 11


def test_plot():
    y_train = np.random.randn(1000)
    p_train = np.random.randn(1000)

    y_valid = np.random.random(1000)
    p_valid = np.random.random(1000)

    # Check if plotting raises errors
    plot_duration_histograms(y_train, p_train, y_valid, p_valid)
