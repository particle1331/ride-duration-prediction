import os
import math

import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import DictVectorizer

from ride_duration.utils import convert_to_dict, plot_duration_histograms
from ride_duration.config import MODEL_DIR, DATASET_DIR, config
from ride_duration.processing import prepare_features


def test_pipeline_training():
    """Run training pipeline."""

    # Load raw dataset
    train = pd.read_parquet(DATASET_DIR / config.TRAIN_SAMPLE)
    valid = pd.read_parquet(DATASET_DIR / config.VALID_SAMPLE)

    # Note: Feature selection defaults to config.FEATURES
    X_train, y_train = prepare_features(train, train=True)
    X_valid, y_valid = prepare_features(valid, train=True)

    # Fit model pipeline (all trainable here)
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

    # Persist trained model
    model_path = MODEL_DIR / "model_pipe.pkl"
    joblib.dump(pipe, model_path)

    assert os.path.exists(model_path)
    assert math.isclose(mse_train,  9.838799799829626, abs_tol=0.1)
    assert math.isclose(mse_valid, 10.499110710362512, abs_tol=0.1)


def test_pipeline_inference():
    """Running inference pipeline. Same transforms as above test."""

    data = pd.read_parquet(DATASET_DIR / config.VALID_SAMPLE)
    model = joblib.load(MODEL_DIR / config.MODEL_SAMPLE)
    X = prepare_features(data)[0]
    p = model.predict(X)

    assert math.isclose(p.mean(), 16.69474798410946, abs_tol=0.1)


def test_plot():
    y_train = np.random.randn(100)
    p_train = np.random.randn(100)

    y_valid = np.random.random(100)
    p_valid = np.random.random(100)

    # Check if plotting raises errors
    plot_duration_histograms(y_train, p_train, y_valid, p_valid)
