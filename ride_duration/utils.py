import seaborn as sns
import matplotlib.pyplot as plt

from ride_duration.config import config


def filter_ride_duration(df):
    """Create target column and filter outliers."""

    df[config.TARGET] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df[config.TARGET] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= config.TARGET_MIN) & (df.duration <= config.TARGET_MAX)]

    return df


def convert_to_dict(df):
    """Convert dataframe to feature dicts."""
    return df.to_dict(orient='records')


def plot_duration_histograms(y_train, yp_train, y_valid, yp_valid):
    """Plot true and prediction distributions of ride duration."""

    # fmt: off
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    pred_config = dict(label='pred', color='C0', stat='density', kde=True)
    true_config = dict(label='true', color='C1', stat='density', kde=True)

    ax[0].set_title("train")
    sns.histplot(yp_train, ax=ax[0], **pred_config)
    sns.histplot(y_train,  ax=ax[0], **true_config)

    ax[1].set_title("valid")
    sns.histplot(yp_valid, ax=ax[1], **pred_config)
    sns.histplot(y_valid,  ax=ax[1], **true_config)

    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()

    return fig
