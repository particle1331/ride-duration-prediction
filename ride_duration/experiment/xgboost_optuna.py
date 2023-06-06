import os

import mlflow
import optuna
from xgboost import XGBRegressor

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
mlflow.xgboost.autolog()
data = data_dict(debug=int(os.environ["DEBUG"]))


def run(params: dict, pudo: int):
    with mlflow.start_run():
        # Preprocessing
        train = data["train_data"]
        valid = data["valid_data"]
        X_train, y_train = preprocess(train, target=True, filter_target=True)
        X_valid, y_valid = preprocess(valid, target=True, filter_target=True)

        # Fit feature pipe
        transforms = [add_pudo_column, feature_selector] if pudo else []
        feature_pipe = feature_pipeline(transforms)
        X_train = feature_pipe.fit_transform(X_train)
        X_valid = feature_pipe.transform(X_valid)

        # Fit model
        model = XGBRegressor(early_stopping_rounds=50, **params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
        )

        # Default mlflow logs
        MODEL_TAG = "xgboost"
        args = [model, MODEL_TAG, data, X_train, y_train, X_valid, y_valid]
        logs = mlflow_default_logging(*args)

        # Logging feature pipe, use pudo
        mlflow.log_param("pudo", pudo)
        mlflow.sklearn.log_model(feature_pipe, "feature_pipe")

    return logs["rmse_valid"]


def objective(trial):
    # fmt: off
    params = {
        "max_depth":        trial.suggest_int("max_depth", 4, 100),
        "n_estimators":     trial.suggest_int("n_estimators", 1, 10, step=1) * 100,
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 1, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 1000, log=True),
        "objective": "reg:squarederror",
        "seed": 42,
    }

    return run(params, pudo=trial.suggest_categorical("pudo", [0, 1]))


if __name__ == "__main__":
    import sys

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=int(sys.argv[1]))
