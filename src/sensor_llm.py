from sklearn.ensemble import RandomForestClassifier

from sensor_processing import (
    create_sensor_features,
    load_labels
)

from llm import analyze_activity_sequence


print("======================================")
print(" SENSOR + ML + LLM PIPELINE")
print("======================================")


# -----------------------------
# Load sensor data
# -----------------------------

print("\nCreating training sensor features...")

X_train = create_sensor_features("train")
y_train = load_labels("train")

print("Training data:", X_train.shape)


print("\nCreating testing sensor features...")

X_test = create_sensor_features("test")

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

print("Model trained successfully.")


# -----------------------------
# Predict test sensor data
# -----------------------------

print("\nPredicting activities...")

predictions = model.predict(X_test)


# -----------------------------
# Create activity sequence
# -----------------------------

cleaned_sequence = []

previous_activity = None

for activity in predictions:

    if activity != previous_activity:

        cleaned_sequence.append(activity)

        previous_activity = activity


# -----------------------------
# Display sequence
# -----------------------------

print("\nActivity sequence:")

for activity in cleaned_sequence:

    print(activity)


# -----------------------------
# Send sequence to LLM
# -----------------------------

print("\nSending activity sequence to LLM...")

llm_result = analyze_activity_sequence(
    cleaned_sequence,
    "Sensor Test Data"
)


# -----------------------------
# Display LLM result
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
    "\nActivity sequence saved to "
    "sensor_activity_sequence.txt"
)