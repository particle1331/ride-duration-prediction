import math

from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import DictVectorizer

from ride_duration.utils import convert_to_dict
from ride_duration.processing import preprocess


def test_pipeline_training(train, valid):
    """Run training pipeline."""

    X_train, y_train = preprocess(train, train=True)
    X_valid, y_valid = preprocess(valid, train=True)

    # Fit model pipeline (stateful transforms + model)
    pipe = make_pipeline(
        FunctionTransformer(convert_to_dict),
        DictVectorizer(),
        LinearRegression(),
    )

    pipe.fit(X_train, y_train)

    # Check performance
    yp_train = pipe.predict(X_train)
    yp_valid = pipe.predict(X_valid)

    mse_train = mean_squared_error(y_train, yp_train, squared=False)
    mse_valid = mean_squared_error(y_valid, yp_valid, squared=False)

    assert mse_train <= 12.0
    assert mse_valid <= 15.0


def test_pipeline_inference(model, valid):
    """Running inference pipeline. Same transforms as above test."""

    X = preprocess(valid, train=False)
    yp = model.predict(X)

    assert math.isclose(yp.mean(), 16.0, abs_tol=5.0)
