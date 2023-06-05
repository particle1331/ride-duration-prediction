import time

import joblib
import mlflow
import optuna
from toolz import compose
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

from ride_duration.utils import plot_duration_histograms
from ride_duration.config import config
from ride_duration.processing import preprocess
from ride_duration.experiment.utils import (
    fixtures,
    add_pudo_column,
    feature_pipeline,
    feature_selector,
    setup_experiment,
)

setup_experiment()
data = fixtures()
mlflow.xgboost.autolog()


def run(params, pudo: bool):
    with mlflow.start_run():
        train_data = data["train_data"]
        valid_data = data["valid_data"]
        train_data_path = data["train_data_path"]
        valid_data_path = data["valid_data_path"]

        # Preprocessing
        X_train, y_train = preprocess(train_data, target=True, filter_target=True)
        X_valid, y_valid = preprocess(valid_data, target=True, filter_target=True)

        # Feature engineering + selection
        transforms = []
        if pudo:
            transforms.extend([add_pudo_column, feature_selector])

        # Fit feature pipe
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

        # Compute metrics
        start_time = time.time()
        yp_train = model.predict(X_train)
        yp_valid = model.predict(X_valid)
        predict_time = time.time() - start_time

        rmse_train = mean_squared_error(y_train, yp_train, squared=False)
        rmse_valid = mean_squared_error(y_valid, yp_valid, squared=False)

        fig = plot_duration_histograms(y_train, yp_train, y_valid, yp_valid)

        # MLflow logging
        mlflow.set_tag("author", "particle")
        mlflow.set_tag("model", "xgboost")

        mlflow.log_param("pudo", pudo)
        mlflow.log_param("train_data_path", train_data_path)
        mlflow.log_param("valid_data_path", valid_data_path)

        mlflow.log_metric("rmse_train", rmse_train)
        mlflow.log_metric("rmse_valid", rmse_valid)

        mlflow.log_metric(
            "inference_time", predict_time / (len(yp_train) + len(yp_valid))
        )

        mlflow.log_figure(fig, "plot.svg")

        # Log feature pipeline as artifact
        mlflow.sklearn.log_model(feature_pipe, "feature_pipe")

    return rmse_valid


def objective(trial, pudo: bool):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 1, 10, step=1) * 100,
        "max_depth": trial.suggest_int("max_depth", 4, 100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 0.1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 0.1, log=True),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 0.1, 1000, log=True
        ),
        "objective": "reg:squarederror",
        "seed": 42,
    }

    return run(params, pudo)


if __name__ == "__main__":
    import sys
    from functools import partial

    N_TRIALS = int(sys.argv[1])
    USE_PUDO = int(sys.argv[2])

    study = optuna.create_study(direction="minimize")
    study.optimize(partial(objective, pudo=USE_PUDO), n_trials=N_TRIALS)
