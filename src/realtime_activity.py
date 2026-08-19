import time
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier

from sensor_processing import (
    create_sensor_features,
    load_labels
)


print("======================================")
print(" REAL-TIME ACTIVITY MONITOR")
print("======================================")


# -----------------------------
# Load training data
# -----------------------------

print("\nLoading training sensor data...")

X_train = create_sensor_features("train")
y_train = load_labels("train")


# -----------------------------
# Train model
# -----------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully.")


# -----------------------------
# Load test sensor data
# -----------------------------

print("\nLoading sensor stream...")

X_test = create_sensor_features("test")

print("Sensor stream:", X_test.shape)


# -----------------------------
# Activity timeline
# -----------------------------

activity_timeline = []

previous_activity = None

print("\n======================================")
print(" LIVE ACTIVITY")
print("======================================\n")


# -----------------------------
# Simulate real-time stream
# -----------------------------

for i in range(len(X_test)):

    sensor_window = X_test.iloc[[i]]

    prediction = model.predict(
        sensor_window
    )[0]

    # Record only activity changes
    if prediction != previous_activity:

        timestamp = datetime.now().strftime(
            "%H:%M:%S"
        )

        activity_timeline.append({
            "time": timestamp,
            "activity": prediction
        })

        print(
            f"[{timestamp}] {prediction}"
        )

        previous_activity = prediction

    # Simulate sensor arrival
    time.sleep(0.05)


# -----------------------------
# Save timeline
# -----------------------------

with open(
    "activity_timeline.txt",
    "w"
) as file:

    for event in activity_timeline:

        file.write(
            f"{event['time']} - "
            f"{event['activity']}\n"
        )


print("\n======================================")
print(" ACTIVITY MONITORING COMPLETE")
print("======================================")

print(
    f"\nDetected {len(activity_timeline)} "
    "activity transitions."
)

print(
    "\nTimeline saved to "
    "activity_timeline.txt"
)