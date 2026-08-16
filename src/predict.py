from load_data import load_test_data, load_subject_test
from llm import analyze_activity_sequence


activity_map = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}


def run_prediction(model):

    # -----------------------------
    # Load test data
    # -----------------------------

    X_test, _ = load_test_data()

    subject_test = load_subject_test()


    # -----------------------------
    # Predict
    # -----------------------------

    predictions = model.predict(X_test)


    # -----------------------------
    # Select Subject 2
    # -----------------------------

    subject_id = 2

    subject_mask = subject_test[0] == subject_id

    subject_predictions = predictions[
        subject_mask.values
    ]


    # -----------------------------
    # Convert predictions to names
    # -----------------------------

    activity_sequence = [
        activity_map[p]
        for p in subject_predictions
    ]


    # -----------------------------
    # Remove consecutive duplicates
    # -----------------------------

    cleaned_sequence = []

    previous_activity = None

    for activity in activity_sequence:

        if activity != previous_activity:

            cleaned_sequence.append(activity)

            previous_activity = activity


    # -----------------------------
    # Display sequence
    # -----------------------------

    print(f"\nSubject: {subject_id}")

    print("\nActivity sequence:")

    for activity in cleaned_sequence:
        print(activity)


    # -----------------------------
    # Send sequence to LLM
    # -----------------------------

    print("\nSending sequence to LLM...")

    llm_result = analyze_activity_sequence(
        cleaned_sequence,
        subject_id
    )


    # -----------------------------
    # Display LLM result
    # -----------------------------

    print("\n==============================")
    print("LLM INTERPRETATION")
    print("==============================\n")

    print(llm_result)


    # -----------------------------
    # Save sequence
    # -----------------------------

    with open(
        "activity_sequence.txt",
        "w"
    ) as file:

        for activity in cleaned_sequence:
            file.write(activity + "\n")


    print(
        "\nActivity sequence saved to activity_sequence.txt"
    )


    # Return sequence for visualization
    return cleaned_sequence