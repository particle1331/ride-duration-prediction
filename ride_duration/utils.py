from ride_duration.config.core import config


def filter_ride_duration(df):
    """Create target column and filter outliers."""
    df[config.TARGET] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df[config.TARGET] = df.duration.dt.total_seconds() / 60
    return df[(df.duration >= config.TARGET_MIN) & (df.duration <= config.TARGET_MAX)]


def dict_features(df, features):
    """Convert dataframe to feature dicts."""
    return df[features].to_dict(orient='records')


def preprocess(df):
    """Preprocess data for training."""
    df = df[config.FEATURES]
    df = filter_ride_duration(df)
    df[config.CATEGORICAL] = df[config.CATEGORICAL].astype(str)
    df[config.NUMERICAL] = df[config.NUMERICAL].astype(float)
    return df


def plot_duration_histograms(y_train, p_train, y_valid, p_valid):
    """Plot true and prediction distributions of ride duration."""

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    sns.histplot(p_train, ax=ax[0], label='pred', color='C0', stat='density', kde=True)
    sns.histplot(y_train, ax=ax[0], label='true', color='C1', stat='density', kde=True)
    ax[0].set_title("Train")
    ax[0].legend()

    sns.histplot(p_valid, ax=ax[1], label='pred', color='C0', stat='density', kde=True)
    sns.histplot(y_valid, ax=ax[1], label='true', color='C1', stat='density', kde=True)
    ax[1].set_title("Valid")
    ax[1].legend()

    fig.tight_layout()
