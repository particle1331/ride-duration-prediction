from pathlib import Path

import mlflow
import pandas as pd
from toolz import compose
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import DictVectorizer

from ride_duration.config import config

ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / "data"
EXPERIMENT_NAME = "nyc-green-taxi"
TRACKING_URI = "http://127.0.0.1:5001"


def setup_experiment():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def fixtures():
    train_data_path = DATA_DIR / "green_tripdata_2021-01.parquet"
    valid_data_path = DATA_DIR / "green_tripdata_2021-02.parquet"
    train_data = pd.read_parquet(train_data_path)
    valid_data = pd.read_parquet(valid_data_path)

    return {
        "train_data_path": train_data_path,
        "valid_data_path": valid_data_path,
        "train_data": train_data,
        "valid_data": valid_data,
    }


def add_pudo_column(df):
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]
    return df


def feature_selector(df):
    if "PU_DO" in df.columns:
        df = df[["PU_DO"] + config.NUM_FEATURES]
    return df


def convert_to_dict(df):
    return df.to_dict(orient="records")


def feature_pipeline(transforms: list):
    def preprocessor(df):
        return compose(*transforms[::-1])(df)

    return make_pipeline(
        FunctionTransformer(preprocessor),
        FunctionTransformer(convert_to_dict),
        DictVectorizer(),
    )
