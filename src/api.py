from flask import Flask, request, jsonify, render_template
import os
import threading
from collections import deque

import numpy as np
import pandas as pd
import joblib

from llm import analyze_activity_sequence


# =========================================
# FLASK APP
# =========================================

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static",
    static_url_path="/static"
)


# =========================================
# PROJECT PATHS
# =========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = r"C:\Users\Admin\OneDrive\Documents\GitHub\human-activity-recognition-llm\sensor_model.pkl"

SEQUENCE_FILE = os.path.join(
    BASE_DIR,
    "sensor_activity_sequence.txt"
)


# =========================================
# LOAD RANDOM FOREST MODEL
# =========================================

print("\nLoading Random Forest model...")

if os.path.exists(MODEL_PATH):

    model = joblib.load(MODEL_PATH)

    print(
        "Random Forest model loaded successfully."
    )

else:

    model = None

    print(
        "WARNING: sensor_model.pkl not found."
    )


# =========================================
# LIVE SENSOR BUFFER
# =========================================

# UCI HAR uses windows of 128 samples.

sensor_buffer = deque(
    maxlen=128
)


# =========================================
# ACTIVITY INFORMATION
# =========================================

latest_activity = "Waiting..."

prediction_count = 0


# =========================================
# LIVE ACTIVITY SEQUENCE
# =========================================

# Only store an activity when
# the prediction changes.

live_activity_sequence = []

last_activity = None


# =========================================
# LLM SETTINGS
# =========================================

# Run LLM after every 10 NEW activity changes.

LLM_SEQUENCE_LENGTH = 10

latest_llm_result = (
    "LLM analysis will appear after "
    "10 activity changes."
)

# Number of activities already processed
# by the last completed LLM analysis.

last_llm_sequence_length = 0

# True while Ollama is running.

llm_running = False

# Prevent multiple LLM threads.

llm_lock = threading.Lock()


# =========================================
# DASHBOARD
# =========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================
# SENSOR TEST PAGE
# =========================================

@app.route("/sensor-page")
def sensor_page():

    return render_template(
        "sensor.html"
    )


# =========================================
# CREATE 30 FEATURES
# =========================================

def create_live_features(samples):

    """
    Convert 128 live sensor samples
    into 30 statistical features.

    6 sensor signals × 5 features = 30 features.

    Features for each signal:

    1. Mean
    2. Standard deviation
    3. Minimum
    4. Maximum
    5. Mean square / energy
    """

    data = pd.DataFrame(samples)

    sensor_columns = [

        "acc_x",
        "acc_y",
        "acc_z",

        "gyro_x",
        "gyro_y",
        "gyro_z"

    ]

    feature_values = []


    # =====================================
    # EXTRACT FEATURES
    # =====================================

    for column in sensor_columns:

        values = (
            data[column]
            .astype(float)
            .values
        )

        # Mean
        feature_values.append(
            np.mean(values)
        )

        # Standard deviation
        feature_values.append(
            np.std(values)
        )

        # Minimum
        feature_values.append(
            np.min(values)
        )

        # Maximum
        feature_values.append(
            np.max(values)
        )

        # Mean square / energy
        feature_values.append(
            np.mean(values ** 2)
        )


    # =====================================
    # FEATURE NAMES
    # =====================================

    feature_names = [

        "acc_x_mean",
        "acc_x_std",
        "acc_x_min",
        "acc_x_max",
        "acc_x_energy",

        "acc_y_mean",
        "acc_y_std",
        "acc_y_min",
        "acc_y_max",
        "acc_y_energy",

        "acc_z_mean",
        "acc_z_std",
        "acc_z_min",
        "acc_z_max",
        "acc_z_energy",

        "gyro_x_mean",
        "gyro_x_std",
        "gyro_x_min",
        "gyro_x_max",
        "gyro_x_energy",

        "gyro_y_mean",
        "gyro_y_std",
        "gyro_y_min",
        "gyro_y_max",
        "gyro_y_energy",

        "gyro_z_mean",
        "gyro_z_std",
        "gyro_z_min",
        "gyro_z_max",
        "gyro_z_energy"

    ]


    return pd.DataFrame(
        [feature_values],
        columns=feature_names
    )


# =========================================
# SAVE ACTIVITY
# =========================================

def save_activity(activity):

    try:

        with open(
            SEQUENCE_FILE,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                activity + "\n"
            )

    except Exception as error:

        print(
            "Could not save activity:",
            error
        )


# =========================================
# RUN LLM ANALYSIS
# =========================================

def run_llm_analysis():

    global latest_llm_result
    global last_llm_sequence_length
    global llm_running


    # =====================================
    # CHECK SEQUENCE LENGTH
    # =====================================

    if len(live_activity_sequence) < LLM_SEQUENCE_LENGTH:

        return


    # =====================================
    # CHECK FOR 10 NEW ACTIVITIES
    # =====================================

    if (
        len(live_activity_sequence)
        - last_llm_sequence_length
        < LLM_SEQUENCE_LENGTH
    ):

        return


    # =====================================
    # PREVENT MULTIPLE REQUESTS
    # =====================================

    with llm_lock:

        if llm_running:

            print(
                "LLM is already running. Skipping request."
            )

            return

        # Mark as running BEFORE starting thread.
        # This prevents duplicate threads.

        llm_running = True


    # =====================================
    # COPY SEQUENCE
    # =====================================

    sequence_for_llm = (
        live_activity_sequence.copy()
    )


    print(
        "\n================================"
    )

    print(
        " SENDING ACTIVITY SEQUENCE TO LLM"
    )

    print(
        "================================"
    )

    print(
        "Sequence:",
        sequence_for_llm
    )


    # =====================================
    # BACKGROUND WORKER
    # =====================================

    def llm_worker():

        global latest_llm_result
        global last_llm_sequence_length
        global llm_running


        try:

            print(
                "\n================================"
            )

            print(
                " BACKGROUND LLM ANALYSIS STARTED"
            )

            print(
                "================================"
            )


            # =================================
            # CALL OLLAMA
            # =================================

            result = analyze_activity_sequence(
                sequence_for_llm,
                "Live Phone"
            )


            # =================================
            # SAVE RESULT
            # =================================

            latest_llm_result = result


            # =================================
            # MARK SEQUENCE AS PROCESSED
            # =================================

            last_llm_sequence_length = (
                len(sequence_for_llm)
            )


            print(
                "\n================================"
            )

            print(
                " BACKGROUND LLM ANALYSIS COMPLETE"
            )

            print(
                "================================"
            )

            print(
                latest_llm_result
            )


        except Exception as error:

            print(
                "\n================================"
            )

            print(
                " LLM ERROR"
            )

            print(
                "================================"
            )

            print(
                error
            )


            latest_llm_result = (
                "LLM analysis failed: "
                + str(error)
            )


        finally:

            # =================================
            # MARK LLM AS FINISHED
            # =================================

            llm_running = False


    # =====================================
    # START BACKGROUND THREAD
    # =====================================

    thread = threading.Thread(
        target=llm_worker,
        daemon=True
    )

    thread.start()


# =========================================
# PREDICT ACTIVITY
# =========================================

def predict_activity():

    global latest_activity
    global prediction_count
    global last_activity


    # =====================================
    # NEED 128 SAMPLES
    # =====================================

    if len(sensor_buffer) < 128:

        return None


    # =====================================
    # CREATE 30 FEATURES
    # =====================================

    features = create_live_features(
        list(sensor_buffer)
    )


    # =====================================
    # CHECK MODEL
    # =====================================

    if model is None:

        print(
            "Cannot predict: model not loaded."
        )

        return None


    # =====================================
    # RANDOM FOREST PREDICTION
    # =====================================

    prediction = model.predict(
        features
    )[0]


    latest_activity = str(
        prediction
    )

    prediction_count += 1


    print(
        "\n================================"
    )

    print(
        " LIVE ACTIVITY PREDICTION"
    )

    print(
        "================================"
    )

    print(
        "Prediction:",
        latest_activity
    )

    print(
        "Samples:",
        len(sensor_buffer)
    )

    print(
        "Prediction number:",
        prediction_count
    )


    # =====================================
    # ACTIVITY CHANGE DETECTION
    # =====================================

    if latest_activity != last_activity:

        # Add only when activity changes.

        live_activity_sequence.append(
            latest_activity
        )


        # Update previous activity.

        last_activity = latest_activity


        print(
            "Activity changed!"
        )

        print(
            "Live sequence:",
            live_activity_sequence
        )


        # =================================
        # SAVE ACTIVITY
        # =================================

        save_activity(
            latest_activity
        )


        # =================================
        # CHECK WHETHER LLM SHOULD RUN
        # =================================

        if (
            len(live_activity_sequence)
            - last_llm_sequence_length
            >= LLM_SEQUENCE_LENGTH
        ):

            run_llm_analysis()


    return latest_activity


# =========================================
# RECEIVE SENSOR DATA
# =========================================

@app.route(
    "/sensor",
    methods=["POST"]
)
def receive_sensor_data():

    try:

        data = request.get_json()


        # =====================================
        # CHECK REQUEST
        # =====================================

        if not data:

            return jsonify({

                "error":
                    "No sensor data received"

            }), 400


        # =====================================
        # GET ACCELEROMETER
        # =====================================

        accelerometer = data.get(
            "accelerometer",
            {}
        )


        # =====================================
        # GET GYROSCOPE
        # =====================================

        gyroscope = data.get(
            "gyroscope",
            {}
        )


        # =====================================
        # EXTRACT VALUES
        # =====================================

        acc_x = accelerometer.get("x")
        acc_y = accelerometer.get("y")
        acc_z = accelerometer.get("z")

        gyro_x = gyroscope.get("x")
        gyro_y = gyroscope.get("y")
        gyro_z = gyroscope.get("z")


        values = [

            acc_x,
            acc_y,
            acc_z,

            gyro_x,
            gyro_y,
            gyro_z

        ]


        # =====================================
        # CHECK MISSING VALUES
        # =====================================

        if any(
            value is None
            for value in values
        ):

            return jsonify({

                "error":
                    "Sensor values are missing",

                "received":
                    data

            }), 400


        # =====================================
        # CONVERT TO FLOAT
        # =====================================

        sample = {

            "acc_x":
                float(acc_x),

            "acc_y":
                float(acc_y),

            "acc_z":
                float(acc_z),

            "gyro_x":
                float(gyro_x),

            "gyro_y":
                float(gyro_y),

            "gyro_z":
                float(gyro_z)

        }


        # =====================================
        # ADD SAMPLE TO BUFFER
        # =====================================

        sensor_buffer.append(
            sample
        )


        print(
            "\nReceived sensor sample:"
        )

        print(
            sample
        )

        print(
            "Buffer:",
            len(sensor_buffer),
            "/ 128"
        )


        # =====================================
        # PREDICT WHEN BUFFER IS FULL
        # =====================================

        activity = None


        if len(sensor_buffer) == 128:

            activity = predict_activity()


        # =====================================
        # RETURN RESPONSE
        # =====================================

        return jsonify({

            "status":
                "success",

            "message":
                "Sensor data received",

            "buffer_size":
                len(sensor_buffer),

            "required_samples":
                128,

            "current_activity":
                latest_activity,

            "prediction":
                activity,

            "llm_running":
                llm_running

        })


    except Exception as error:

        print(
            "\nERROR receiving sensor data:"
        )

        print(
            error
        )


        return jsonify({

            "status":
                "error",

            "message":
                str(error)

        }), 500


# =========================================
# ACTIVITY API
# =========================================

@app.route(
    "/api/activity"
)
def get_activity():

    sequence = []


    # =====================================
    # READ SAVED ACTIVITY SEQUENCE
    # =====================================

    if os.path.exists(
        SEQUENCE_FILE
    ):

        try:

            with open(
                SEQUENCE_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                sequence = [

                    line.strip()

                    for line in file

                    if line.strip()

                ]


        except Exception as error:

            print(
                "Could not read sequence:",
                error
            )


    # =====================================
    # CURRENT ACTIVITY
    # =====================================

    current = latest_activity


    if (
        sequence
        and
        current == "Waiting..."
    ):

        current = sequence[-1]


    # =====================================
    # DASHBOARD RESPONSE
    # =====================================

    return jsonify({

        "current_activity":
            current,

        "sequence":
            sequence,

        "live_sequence":
            live_activity_sequence,

        "accuracy":
            82.08,

        "features":
            30,

        "buffer_size":
            len(sensor_buffer),

        "required_samples":
            128,

        "prediction_count":
            prediction_count,

        "llm_running":
            llm_running,

        "llm_interpretation":
            latest_llm_result

    })


# =========================================
# RUN FLASK
# =========================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )