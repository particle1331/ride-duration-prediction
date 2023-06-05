from ride_duration.config import config
from ride_duration.processing import preprocess


def test_preprocess_train(train):
    orig_shape = train.shape
    X, y = preprocess(train, train=True)

    assert len(X) == len(y)
    assert X.shape[0] <= orig_shape[0]
    assert X.shape[1] == len(config.FEATURES)
    assert sorted(list(X.columns)) == sorted(config.FEATURES)
    assert config.TARGET not in X.columns


def test_preprocess_infer(train):
    orig_shape = train.shape
    X = preprocess(train, train=False)

    assert sorted(list(X.columns)) == sorted(config.FEATURES)
    assert X.shape[0] == orig_shape[0]
    assert X.shape[1] == len(config.FEATURES)
    assert config.TARGET not in X.columns
