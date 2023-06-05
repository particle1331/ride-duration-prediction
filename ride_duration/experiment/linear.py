import time

import mlflow
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from ride_duration.utils import plot_duration_histograms
from ride_duration.processing import preprocess
from ride_duration.experiment.utils import fixtures, feature_pipeline, setup_experiment

setup_experiment()
data = fixtures()


with mlflow.start_run():
    train_data = data["train_data"]
    valid_data = data["valid_data"]
    train_data_path = data["train_data_path"]
    valid_data_path = data["valid_data_path"]

    # Preprocessing
    X_train, y_train = preprocess(train_data, target=True, filter_target=True)
    X_valid, y_valid = preprocess(valid_data, target=True, filter_target=True)

    # Fit feature pipe
    feature_pipe = feature_pipeline()
    X_train = feature_pipe.fit_transform(X_train)
    X_valid = feature_pipe.transform(X_valid)

    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

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
    mlflow.set_tag("model", "linear")

    mlflow.log_param("train_data_path", train_data_path)
    mlflow.log_param("valid_data_path", valid_data_path)

    mlflow.log_metric("rmse_train", rmse_train)
    mlflow.log_metric("rmse_valid", rmse_valid)

    mlflow.log_metric("inference_time", predict_time / (len(yp_train) + len(yp_valid)))

    mlflow.log_figure(fig, "plot.svg")

    # Log feature pipeline as artifact
    mlflow.sklearn.log_model(feature_pipe, "feature_pipe")
