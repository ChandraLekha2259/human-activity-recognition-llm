import pandas as pd


DATA_PATH = "data/UCI HAR Dataset/"


# -----------------------------
# Load feature names
# -----------------------------

features = pd.read_csv(
    DATA_PATH + "features.txt",
    sep=r"\s+",
    header=None
)

feature_names = features[1].values


# -----------------------------
# Load training data
# -----------------------------

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


# Assign meaningful feature names
X_train.columns = feature_names


# -----------------------------
# Load activity names
# -----------------------------

activities = pd.read_csv(
    DATA_PATH + "activity_labels.txt",
    sep=r"\s+",
    header=None
)

activity_map = dict(
    zip(
        activities[0],
        activities[1]
    )
)


# Convert activity IDs to names
y_train[0] = y_train[0].map(activity_map)


# -----------------------------
# Dataset information
# -----------------------------

print("Training data shape:")
print(X_train.shape)

print("\nNumber of features:")
print(len(feature_names))

print("\nActivity distribution:")
print(y_train[0].value_counts())

print("\nMissing values:")
print(X_train.isnull().sum().sum())

print("\nFirst 5 activity labels:")
print(y_train.head())