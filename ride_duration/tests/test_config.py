import os

from ride_duration.config import DATASET_DIR, config


def test():
    assert os.path.exists(DATASET_DIR / config.TRAIN_SAMPLE)
    assert os.path.exists(DATASET_DIR / config.VALID_SAMPLE)
    assert set(config.CATEGORICAL + config.NUMERICAL) <= set(config.FEATURES)
