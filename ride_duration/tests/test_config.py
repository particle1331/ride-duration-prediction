import os

from ride_duration.config import MODEL_DIR, DATASET_DIR, config


def test():
    assert os.path.exists(DATASET_DIR / config.TRAIN_SAMPLE)
    assert os.path.exists(DATASET_DIR / config.VALID_SAMPLE)
    assert os.path.exists(MODEL_DIR   / config.MODEL_SAMPLE)
    assert set(config.CAT_FEATURES + config.NUM_FEATURES) <= set(config.FEATURES)
