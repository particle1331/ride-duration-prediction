import math

from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import DictVectorizer

from ride_duration.utils import convert_to_dict
from ride_duration.processing import prepare_features


def test_pipeline_training(train, valid):
    """Run training pipeline."""

    X_train, y_train = prepare_features(train, train=True)
    X_valid, y_valid = prepare_features(valid, train=True)

    # Fit model pipeline (transforms (trainable) + model)
    pipe = make_pipeline(
        FunctionTransformer(convert_to_dict), DictVectorizer(), LinearRegression()
    )

    pipe.fit(X_train, y_train)

    # Check performance
    pred_train = pipe.predict(X_train)
    pred_valid = pipe.predict(X_valid)

    mse_train = mean_squared_error(y_train, pred_train, squared=False)
    mse_valid = mean_squared_error(y_valid, pred_valid, squared=False)

    assert mse_train <= 12.0
    assert mse_valid <= 15.0


def test_pipeline_inference(model, valid):
    """Running inference pipeline. Same transforms as above test."""

    X = prepare_features(valid)[0]
    pred = model.predict(X)

    assert math.isclose(pred.mean(), 16.0, abs_tol=5.0)
