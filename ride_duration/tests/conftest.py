import joblib
import pandas as pd
import pytest

from ride_duration.config import MODEL_DIR, DATASET_DIR, config


@pytest.fixture
def train():
    return pd.read_parquet(DATASET_DIR / config.TRAIN_SAMPLE)


@pytest.fixture
def valid():
    return pd.read_parquet(DATASET_DIR / config.VALID_SAMPLE)


@pytest.fixture
def model():
    return joblib.load(MODEL_DIR / config.MODEL_SAMPLE)
