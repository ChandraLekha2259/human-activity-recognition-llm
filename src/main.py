from load_data import load_training_data, load_test_data
from model import train_model
from evaluate import evaluate_model
from predict import run_prediction
from visualize import visualize_activity_sequence


def main():

    print("======================================")
    print(" HUMAN ACTIVITY RECOGNITION + LLM")
    print("======================================")


    # -----------------------------
    # Load data
    # -----------------------------

    X_train, y_train = load_training_data()

    X_test, y_test = load_test_data()

    print("\nTraining data:", X_train.shape)
    print("Testing data:", X_test.shape)


    # -----------------------------
    # Train model ONCE
    # -----------------------------

    print("\nTraining Random Forest...")

    model = train_model(
        X_train,
        y_train
    )

    print("Model trained successfully.")


    # -----------------------------
    # Evaluate model
    # -----------------------------

    print("\n======================================")
    print(" MODEL EVALUATION")
    print("======================================")

    evaluate_model(
        model,
        X_test,
        y_test
    )


    # -----------------------------
    # Prediction + LLM
    # -----------------------------

    print("\n======================================")
    print(" ACTIVITY PREDICTION + LLM")
    print("======================================")

    activity_sequence = run_prediction(model)


    # -----------------------------
    # Visualization
    # -----------------------------

    print("\n======================================")
    print(" ACTIVITY VISUALIZATION")
    print("======================================")

    visualize_activity_sequence(
        activity_sequence
    )


if __name__ == "__main__":
    main()