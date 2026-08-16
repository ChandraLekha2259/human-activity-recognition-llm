import pandas as pd


DATA_PATH = "data/UCI HAR Dataset/"


def load_training_data():
    X_train = pd.read_csv(
        DATA_PATH + "train/X_train.txt",
        sep=r"\s+",
        header=None
    )

    y_train = pd.read_csv(
        DATA_PATH + "train/y_train.txt",
        sep=r"\s+",
        header=None
    )

    return X_train, y_train


def load_test_data():
    X_test = pd.read_csv(
        DATA_PATH + "test/X_test.txt",
        sep=r"\s+",
        header=None
    )

    y_test = pd.read_csv(
        DATA_PATH + "test/y_test.txt",
        sep=r"\s+",
        header=None
    )

    return X_test, y_test


def load_subject_test():
    return pd.read_csv(
        DATA_PATH + "test/subject_test.txt",
        sep=r"\s+",
        header=None
    )


def load_features():
    return pd.read_csv(
        DATA_PATH + "features.txt",
        sep=r"\s+",
        header=None
    )


def load_activities():
    return pd.read_csv(
        DATA_PATH + "activity_labels.txt",
        sep=r"\s+",
        header=None
    )