import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from load_data import load_test_data


activity_names = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\nAccuracy:", accuracy)

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=activity_names
        )
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nConfusion Matrix:\n")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=activity_names
    )

    disp.plot(
        xticks_rotation=45
    )

    plt.title(
        "Human Activity Recognition - Confusion Matrix"
    )

    plt.tight_layout()

    plt.show()

    return accuracy