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


def test_pipeline():
    # Load and preprocess data
    df = pd.read_parquet(DATASET_DIR / "green_tripdata_2021-01.parquet")
    orig_dataset_size = df.shape[0]
    df = preprocess(df)

    # Feature selection
    SELECTED_FEATURES = config.CATEGORICAL + config.NUMERICAL
    y = df[config.TARGET].values
    X = df[SELECTED_FEATURES]

    # Fit model pipeline
    pipe = make_pipeline(
        FunctionTransformer(convert_to_dict), 
        DictVectorizer(), 
        LinearRegression()
    )
    pipe.fit(X, y)

    # Check performance
    p = pipe.predict(X)
    mse = mean_squared_error(y, p, squared=False)

    assert sorted(list(df.columns)) == sorted(config.FEATURES + [config.TARGET])
    assert df.shape[0] <= orig_dataset_size
    assert mse < 10


def test_plot():
    y_train = np.random.randn(1000)
    p_train = np.random.randn(1000)

    y_valid = np.random.random(1000)
    p_valid = np.random.random(1000)

    # Check if plotting raises errors
    plot_duration_histograms(y_train, p_train, y_valid, p_valid)
