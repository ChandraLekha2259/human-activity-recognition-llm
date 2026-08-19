import time

from sklearn.ensemble import RandomForestClassifier

from sensor_processing import (
    create_sensor_features,
    load_labels
)


print("======================================")
print(" REAL-TIME SENSOR SIMULATION")
print("======================================")


# -----------------------------
# Load training sensor features
# -----------------------------

print("\nLoading training data...")

X_train = create_sensor_features("train")
y_train = load_labels("train")

print("Training data:", X_train.shape)


# -----------------------------
# Train model
# -----------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("Model trained successfully.")


# -----------------------------
# Load test sensor data
# -----------------------------

print("\nLoading test sensor features...")

X_test = create_sensor_features("test")

print("Test data:", X_test.shape)


# -----------------------------
# Simulate real-time input
# -----------------------------

print("\n======================================")
print(" STARTING SENSOR STREAM")
print("======================================\n")

previous_activity = None

for i in range(len(X_test)):

    # Take one sensor window
    sensor_window = X_test.iloc[[i]]

    # Predict activity
    prediction = model.predict(
        sensor_window
    )[0]

    # Display only when activity changes
    if prediction != previous_activity:

        print(
            f"Window {i + 1:4d}  →  {prediction}"
        )

        previous_activity = prediction

    # Small delay to simulate live data
    time.sleep(0.05)


print("\n======================================")
print(" SENSOR STREAM FINISHED")
print("======================================")