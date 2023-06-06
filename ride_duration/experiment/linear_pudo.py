import os

import mlflow
from sklearn.linear_model import LinearRegression

from ride_duration.processing import preprocess
from ride_duration.experiment.utils import (
    data_dict,
    add_pudo_column,
    feature_pipeline,
    feature_selector,
    setup_experiment,
    mlflow_default_logging,
)

setup_experiment()
data = data_dict(debug=int(os.environ["DEBUG"]))


with mlflow.start_run():
    # Preprocessing
    X_train, y_train = preprocess(data["train_data"], target=True, filter_target=True)
    X_valid, y_valid = preprocess(data["valid_data"], target=True, filter_target=True)

    # Feature engineering + selection
    transforms = [add_pudo_column, feature_selector]

    # Fit feature pipe
    feature_pipe = feature_pipeline(transforms)
    X_train = feature_pipe.fit_transform(X_train)
    X_valid = feature_pipe.transform(X_valid)

    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # MLflow logging
    MODEL_TAG = "linear-pudo"
    mlflow_default_logging(model, MODEL_TAG, data, X_train, y_train, X_valid, y_valid)
