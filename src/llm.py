import ollama


def analyze_activity_sequence(activity_sequence, subject_id):

    # Convert sequence into readable text
    sequence_text = " → ".join(activity_sequence)

    # Count occurrences in the compressed sequence
    activity_counts = {}

    for activity in activity_sequence:
        activity_counts[activity] = activity_counts.get(activity, 0) + 1

    counts_text = "\n".join(
        f"- {activity}: {count}"
        for activity, count in activity_counts.items()
    )

    prompt = f"""
You are analyzing predictions produced by a Human Activity Recognition
machine learning model.

Subject ID: {subject_id}

Predicted activity sequence:
{sequence_text}

Activity counts:
{counts_text}

Provide exactly these three sections:

1. Main Activities Detected
2. Important Transitions
3. Overall Activity Pattern

STRICT RULES:

- Use the exact activity names provided in the input.
- Do not replace activity names with synonyms.
- Do not invent activities.
- Do not infer the person's location.
- Do not infer the person's intentions.
- Do not infer medical or physical conditions.
- Do not infer whether the person needed assistance.
- Do not describe an activity as "climbing" unless that exact term
  is present in the input.
- Do not use terms such as "sedentary", "ambulatory",
  "mobility assistance", "exercise", or "rest" unless explicitly
  present in the input.
- Do not infer what happened between two consecutive predictions.
- Describe only patterns directly observable from the sequence.
- For transitions, use the exact activity names.

TRANSITION RULES:

- A transition means two DIFFERENT consecutive activities.
- Never report a transition where the source and destination are identical.
- Do NOT report transitions such as:
  STANDING → STANDING
  SITTING → SITTING
  WALKING → WALKING
  or any other activity → the same activity.
- Report each unique transition only once.
- Use the exact format:
  ACTIVITY_A → ACTIVITY_B

IMPORTANT ABOUT COUNTS:

- The activity counts represent occurrences in the compressed sequence.
- They do NOT represent time spent performing an activity.
- Do not use phrases such as:
  "spends more time",
  "spends equal time",
  "duration",
  "for a long period",
  or "proportion of time".
- You may say that an activity occurs more or fewer times only when
  supported by the provided activity counts.

OVERALL PATTERN:

- Describe only patterns directly visible in the sequence.
- Do not claim that an activity lasts for a particular amount of time.
- Do not make assumptions about the person's real-world situation.

Keep the response concise and suitable for a college project report.
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]