from ride_duration.config import config
from ride_duration.processing import preprocess


def test_preprocess_target_filter(train):
    output = preprocess(train, target=True, filter_target=True)
    X, y = output

    assert len(X) == len(y)
    assert X.shape[0] <= train.shape[0]
    assert X.shape[1] == len(config.FEATURES)
    assert sorted(list(X.columns)) == sorted(config.FEATURES)
    assert config.TARGET not in X.columns


def test_preprocess_target_nofilter(train):
    output = preprocess(train, target=True, filter_target=False)
    X, y = output

    assert len(X) == len(y)
    assert X.shape[0] == train.shape[0]
    assert X.shape[1] == len(config.FEATURES)
    assert sorted(list(X.columns)) == sorted(config.FEATURES)
    assert config.TARGET not in X.columns


def test_preprocess_inference(train):
    X = preprocess(train, target=False)

    assert sorted(list(X.columns)) == sorted(config.FEATURES)
    assert X.shape[0] == train.shape[0]
    assert X.shape[1] == len(config.FEATURES)
    assert config.TARGET not in X.columns
