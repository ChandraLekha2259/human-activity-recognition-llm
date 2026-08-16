import matplotlib.pyplot as plt


activity_map = {
    "WALKING": 1,
    "WALKING_UPSTAIRS": 2,
    "WALKING_DOWNSTAIRS": 3,
    "SITTING": 4,
    "STANDING": 5,
    "LAYING": 6
}


def visualize_activity_sequence(activity_sequence):

    # Convert activity names to numerical positions
    y_values = [
        activity_map[activity]
        for activity in activity_sequence
    ]

    # Create timeline
    plt.figure(figsize=(12, 6))

    plt.plot(
        range(1, len(activity_sequence) + 1),
        y_values,
        marker="o"
    )

    plt.yticks(
        list(activity_map.values()),
        list(activity_map.keys())
    )

    plt.xlabel("Activity Transition")
    plt.ylabel("Detected Activity")

    plt.title(
        "Human Activity Recognition - Subject 2"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()