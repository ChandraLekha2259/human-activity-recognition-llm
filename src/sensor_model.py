from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib

from sensor_processing import (
    create_sensor_features,
    load_labels
)


print("======================================")
print(" SENSOR RANDOM FOREST MODEL")
print("======================================")


# -----------------------------
# Create training features
# -----------------------------

print("\nCreating training sensor features...")

X_train = create_sensor_features("train")
y_train = load_labels("train")

print(
    "Training sensor data:",
    X_train.shape
)


# -----------------------------
# Create testing features
# -----------------------------

print("\nCreating testing sensor features...")

X_test = create_sensor_features("test")
y_test = load_labels("test")

print(
    "Testing sensor data:",
    X_test.shape
)


# -----------------------------
# Train Random Forest
# -----------------------------

print("\nTraining Sensor Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("Sensor model trained successfully.")


# -----------------------------
# SAVE MODEL
# -----------------------------

MODEL_PATH = "sensor_model.pkl"

joblib.dump(
    model,
    MODEL_PATH
)

print(
    f"Sensor model saved successfully to: {MODEL_PATH}"
)


# -----------------------------
# Predict test data
# -----------------------------

print("\nPredicting test activities...")

y_pred = model.predict(X_test)


# -----------------------------
# Accuracy
# -----------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==============================")
print(" SENSOR MODEL EVALUATION")
print("==============================")

print(
    "\nAccuracy:",
    accuracy
)


# -----------------------------
# Classification Report
# -----------------------------

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:\n")

print(cm)


print("\n======================================")
print(" MODEL TRAINING COMPLETED")
print("======================================")