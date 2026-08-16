def simplify_sequence(activity_sequence, min_duration=3):
    groups = []

    current_activity = activity_sequence[0]
    count = 1

    for activity in activity_sequence[1:]:

        if activity == current_activity:
            count += 1

        else:
            groups.append((current_activity, count))

            current_activity = activity
            count = 1

    # Add the final group
    groups.append((current_activity, count))

    simplified = []

    for activity, count in groups:
        if count >= min_duration:
            simplified.append(activity)

    return simplified