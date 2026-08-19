import pandas as pd
import numpy as np


BASE_PATH = "Data/UCI HAR Dataset/"


ACTIVITY_MAP = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}


def load_sensor_data(data_type):

    sensor_path = (
        BASE_PATH
        + data_type
        + "/Inertial Signals/"
    )

    # Accelerometer
    acc_x = pd.read_csv(
        sensor_path + f"body_acc_x_{data_type}.txt",
        sep=r"\s+",
        header=None
    )

    acc_y = pd.read_csv(
        sensor_path + f"body_acc_y_{data_type}.txt",
        sep=r"\s+",
        header=None
    )

    acc_z = pd.read_csv(
        sensor_path + f"body_acc_z_{data_type}.txt",
        sep=r"\s+",
        header=None
    )

    # Gyroscope
    gyro_x = pd.read_csv(
        sensor_path + f"body_gyro_x_{data_type}.txt",
        sep=r"\s+",
        header=None
    )

    gyro_y = pd.read_csv(
        sensor_path + f"body_gyro_y_{data_type}.txt",
        sep=r"\s+",
        header=None
    )

    gyro_z = pd.read_csv(
        sensor_path + f"body_gyro_z_{data_type}.txt",
        sep=r"\s+",
        header=None
    )

    return (
        acc_x,
        acc_y,
        acc_z,
        gyro_x,
        gyro_y,
        gyro_z
    )


def extract_features(signal):

    values = signal.values

    return [
        np.mean(values),
        np.std(values),
        np.min(values),
        np.max(values),
        np.mean(values ** 2)
    ]


def create_sensor_features(data_type):

    (
        acc_x,
        acc_y,
        acc_z,
        gyro_x,
        gyro_y,
        gyro_z
    ) = load_sensor_data(data_type)

    sensor_data = [
        acc_x,
        acc_y,
        acc_z,
        gyro_x,
        gyro_y,
        gyro_z
    ]

    feature_rows = []

    for i in range(len(acc_x)):

        row = []

        for sensor in sensor_data:

            window = sensor.iloc[i]

            row.extend(
                extract_features(window)
            )

        feature_rows.append(row)

    feature_names = [
        "acc_x_mean",
        "acc_x_std",
        "acc_x_min",
        "acc_x_max",
        "acc_x_energy",

        "acc_y_mean",
        "acc_y_std",
        "acc_y_min",
        "acc_y_max",
        "acc_y_energy",

        "acc_z_mean",
        "acc_z_std",
        "acc_z_min",
        "acc_z_max",
        "acc_z_energy",

        "gyro_x_mean",
        "gyro_x_std",
        "gyro_x_min",
        "gyro_x_max",
        "gyro_x_energy",

        "gyro_y_mean",
        "gyro_y_std",
        "gyro_y_min",
        "gyro_y_max",
        "gyro_y_energy",

        "gyro_z_mean",
        "gyro_z_std",
        "gyro_z_min",
        "gyro_z_max",
        "gyro_z_energy"
    ]

    return pd.DataFrame(
        feature_rows,
        columns=feature_names
    )


def load_labels(data_type):

    label_path = (
        BASE_PATH
        + data_type
        + f"/y_{data_type}.txt"
    )

    labels = pd.read_csv(
        label_path,
        sep=r"\s+",
        header=None
    )

    return labels[0].map(ACTIVITY_MAP)


if __name__ == "__main__":

    print("Creating training sensor features...")

    X_train = create_sensor_features("train")

    y_train = load_labels("train")

    print(
        "Training sensor features:",
        X_train.shape
    )

    print(
        "Training labels:",
        y_train.shape
    )

    print("\nCreating testing sensor features...")

    X_test = create_sensor_features("test")

    y_test = load_labels("test")

    print(
        "Testing sensor features:",
        X_test.shape
    )

    print(
        "Testing labels:",
        y_test.shape
    )

    print("\nSensor processing completed.")

    print("\nTraining activity distribution:")
    print(y_train.value_counts())

    print("\nTesting activity distribution:")
    print(y_test.value_counts())