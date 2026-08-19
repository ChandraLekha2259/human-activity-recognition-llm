from sklearn.ensemble import RandomForestClassifier

from sensor_processing import (
    create_sensor_features,
    load_labels
)

from llm import analyze_activity_sequence


print("======================================")
print(" SENSOR ACTIVITY + LLM PIPELINE")
print("======================================")


# -----------------------------
# Create sensor features
# -----------------------------

print("\nCreating training sensor features...")

X_train = create_sensor_features("train")
y_train = load_labels("train")

print("Training data:", X_train.shape)


print("\nCreating testing sensor features...")

X_test = create_sensor_features("test")
y_test = load_labels("test")

print("Testing data:", X_test.shape)


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
# Predict activities
# -----------------------------

print("\nPredicting activities...")

predictions = model.predict(X_test)


# -----------------------------
# Select a portion of the
# test sequence
# -----------------------------

# Use the first 100 predictions
# for the LLM demonstration.

activity_sequence = predictions[:100].tolist()


# -----------------------------
# Remove consecutive duplicates
# -----------------------------

cleaned_sequence = []

previous_activity = None

for activity in activity_sequence:

    if activity != previous_activity:

        cleaned_sequence.append(
            activity
        )

        previous_activity = activity


# -----------------------------
# Display activity sequence
# -----------------------------

print("\nActivity sequence:")

for activity in cleaned_sequence:
    print(activity)


# -----------------------------
# Send sequence to LLM
# -----------------------------

print("\nSending sensor predictions to LLM...")

llm_result = analyze_activity_sequence(
    cleaned_sequence,
    "Test Sensor Sequence"
)


# -----------------------------
# Display LLM interpretation
# -----------------------------

print("\n======================================")
print(" LLM INTERPRETATION")
print("======================================\n")

print(llm_result)


# -----------------------------
# Save sequence
# -----------------------------

with open(
    "sensor_activity_sequence.txt",
    "w"
) as file:

    for activity in cleaned_sequence:

        file.write(
            activity + "\n"
        )


print(
    "\nSensor activity sequence saved to "
    "sensor_activity_sequence.txt"
)