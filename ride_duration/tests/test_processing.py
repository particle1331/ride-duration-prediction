import pandas as pd

from ride_duration.config import DATASET_DIR, config
from ride_duration.processing import preprocess, prepare_features


def test_preprocess_train():
    df = pd.read_parquet(DATASET_DIR / config.TRAIN_SAMPLE)
    orig_shape = df.shape
    df = preprocess(df, train=True)

    assert sorted(list(df.columns)) == sorted(config.FEATURES + [config.TARGET])
    assert df.shape[0] <= orig_shape[0]
    assert df.shape[1] == len(config.FEATURES) + 1
    assert config.TARGET in df.columns


def test_preprocess_infer():
    df = pd.read_parquet(DATASET_DIR / config.TRAIN_SAMPLE)
    orig_shape = df.shape
    df = preprocess(df, train=False)

    assert sorted(list(df.columns)) == sorted(config.FEATURES)
    assert df.shape[0] == orig_shape[0]
    assert df.shape[1] == len(config.FEATURES)
    assert config.TARGET not in df.columns


def test_prepare_features():
    df = pd.read_parquet(DATASET_DIR / config.TRAIN_SAMPLE)
    transforms = [lambda df: df[config.NUMERICAL]]
    X_train, y_train = prepare_features(df, transforms, train=True)
    X_infer, y_infer = prepare_features(df, transforms, train=False)

    assert y_infer is None
    assert y_train is not None
    assert X_train.shape[1] == len(config.NUMERICAL)
    assert X_infer.shape[1] == len(config.NUMERICAL)
    assert X_train.shape[0] < X_infer.shape[0]  # Filtered vs. Not filtered
