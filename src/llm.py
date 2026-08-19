import ollama
import os
import time


def analyze_activity_sequence(activity_sequence, subject_id):

    # =========================================
    # CONVERT SEQUENCE INTO READABLE TEXT
    # =========================================

    sequence_text = " → ".join(activity_sequence)


    # =========================================
    # COUNT ACTIVITIES
    # =========================================

    activity_counts = {}

    for activity in activity_sequence:

        activity_counts[activity] = (
            activity_counts.get(activity, 0) + 1
        )


    counts_text = "\n".join(
        f"- {activity}: {count}"
        for activity, count in activity_counts.items()
    )


    # =========================================
    # CALCULATE UNIQUE TRANSITIONS
    # =========================================

    transitions = []

    seen_transitions = set()


    for i in range(len(activity_sequence) - 1):

        source = activity_sequence[i]

        destination = activity_sequence[i + 1]


        # Ignore identical consecutive activities

        if source == destination:
            continue


        transition = (
            source,
            destination
        )


        # Store each transition only once

        if transition not in seen_transitions:

            seen_transitions.add(
                transition
            )

            transitions.append(
                f"{source} → {destination}"
            )


    transitions_text = "\n".join(
        f"- {transition}"
        for transition in transitions
    )


    # =========================================
    # LLM PROMPT
    # =========================================

    prompt = f"""
You are analyzing predictions from a Human Activity Recognition
machine learning model.

Subject ID: {subject_id}

Predicted activity sequence:

{sequence_text}

Your task is to describe ONLY the overall observable pattern
in this sequence.

STRICT RULES:

- Use only activity names that appear in the sequence.
- Use the exact activity names.
- Do not replace activity names with synonyms.
- Do not invent activities.
- Do not infer location.
- Do not infer intentions.
- Do not infer medical or physical conditions.
- Do not infer what happened between activities.
- Do not describe duration or time spent.
- Do not use words such as:
  "exercise",
  "rest",
  "sedentary",
  "ambulatory",
  "climbing",
  "assistance"
  unless those exact words are present in the activity sequence.
- Describe only patterns directly visible in the sequence.
- Do not list the activities separately.
- Do not list transitions.
- Do not provide headings.
- Do not provide bullet points.

Write exactly 2 or 3 concise sentences describing
the overall observable activity pattern.

Sequence:

{sequence_text}
"""


    # =========================================
    # CALL OLLAMA
    # =========================================

    print(
        "\nSENDING ACTIVITY SEQUENCE TO LLM"
    )

    start_time = time.time()


    response = ollama.chat(

        model="llama3.2:3b",

        messages=[

            {
                "role": "user",
                "content": prompt
            }

        ],

        options={

            "temperature": 0,

            "num_predict": 80

        }

    )


    elapsed_time = (
        time.time() - start_time
    )


    print(
        f"LLM RESPONSE RECEIVED IN "
        f"{elapsed_time:.2f} SECONDS"
    )


    llm_pattern = (
        response["message"]["content"]
        .strip()
    )


    # =========================================
    # DISPLAY LLM RESULT
    # =========================================

    print("\nLLM RESULT:")

    print(llm_pattern)


    # =========================================
    # BUILD FINAL INTERPRETATION
    # =========================================

    final_result = (
        "1. Main Activities Detected\n\n"
        + counts_text
        + "\n\n\n"
        + "2. Important Transitions\n\n"
        + transitions_text
        + "\n\n\n"
        + "3. Overall Activity Pattern\n\n"
        + llm_pattern
    )


    # =========================================
    # SAVE INTERPRETATION
    # =========================================

    base_dir = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )


    output_file = os.path.join(
        base_dir,
        "llm_interpretation.txt"
    )


    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            final_result
        )


    # =========================================
    # RETURN FINAL RESULT
    # =========================================

    return final_result