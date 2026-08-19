# human-activity-recognition-llm
I. Introduction
Human Activity Recognition (HAR) is a useful application of sensor-based machine learning because it allows a system to identify what kind of physical activity is being performed from motion data. In this internship, I worked on developing a Human Activity Recognition system using the accelerometer and gyroscope available in a smartphone.
The main idea of the project was to build a complete working pipeline rather than only train a machine-learning model. I first worked with the UCI Human Activity Recognition dataset and used its accelerometer and gyroscope signals to create features for training. A Random Forest classifier was then trained to recognize six activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, and LAYING.
After training the model, I connected it to a Flask backend so that sensor readings could be received from a real smartphone. The phone continuously provides motion information, which is collected into windows of 128 samples. The same type of feature extraction used during training is then applied to the live data before sending it to the Random Forest model.
One additional part of the project was the use of a Large Language Model through Ollama. Instead of asking the LLM to directly classify raw sensor data, I used it to interpret a sequence of activities predicted by the machine-learning model. This makes the output easier for a person to understand, because the system can describe the important activities, transitions, and the overall pattern in natural language.
II. Objectives of Internship Program
The main objectives of my internship work were to understand how sensor-based Human Activity Recognition works and to develop a complete working prototype. The specific objectives were:
1. To understand the use of smartphone accelerometer and gyroscope data for Human Activity Recognition.
2. To study and process the UCI HAR dataset for training and testing the activity-recognition model.
3. To extract useful statistical information from sensor signals and convert it into a fixed set of features.
4. To train a Random Forest classifier for recognizing six different human activities.
5. To save the trained model and use it later for live predictions.
6. To develop a Flask backend capable of receiving sensor data from a smartphone.
7. To create a web interface where live sensor values and activity predictions can be viewed.
8. To maintain a sequence of predicted activities instead of looking at every individual prediction in isolation.
9. To integrate an LLM through Ollama so that the predicted activity sequence could be explained in a more understandable form.
III. Methodology
The project was developed in a series of steps, starting from the dataset and ending with live activity recognition on a smartphone. I kept the live processing pipeline similar to the training pipeline so that the features given to the model during live prediction would have the same meaning as the features used during training.
Dataset and sensor data: The UCI HAR dataset was used for developing the machine-learning part of the system. The dataset contains accelerometer and gyroscope signals recorded for different human activities. The training and testing data were divided into windows of 128 samples.
Feature extraction: Instead of passing all the raw sensor values directly to the Random Forest model, I extracted five simple statistical features from each sensor signal. These were the mean, standard deviation, minimum value, maximum value, and mean square (energy). There are six signals in total: acc_x, acc_y, acc_z, gyro_x, gyro_y, and gyro_z. Therefore, each 128-sample window produces 6 × 5 = 30 features.
Model training: A Random Forest classifier was trained using the extracted training features and their corresponding activity labels. The model was configured with 100 decision trees and a random state of 42. After training, the model was saved as sensor_model.pkl so that it could be loaded by the live application without training again.
Live smartphone sensing: A web page was created to access the smartphone's motion sensors. The page displays the accelerometer and gyroscope readings and sends the collected values to the Flask backend. The backend receives these values through the /sensor endpoint.
Live prediction: The Flask application stores incoming samples in a buffer containing up to 128 samples. Once the buffer is full, the same 30 features are calculated from the live samples. These features are passed to the saved Random Forest model, which produces one of the six activity labels.
Activity sequence: A single prediction can sometimes change because of differences in phone orientation or movement. To make the output more useful, the system records an activity only when the predicted activity changes. This creates a sequence such as SITTING → WALKING → WALKING_UPSTAIRS → LAYING.
LLM interpretation: After 10 activity changes, the activity sequence is sent to an LLM running through Ollama. The LLM does not replace the Random Forest classifier. Instead, it works as an interpretation layer and explains the sequence in terms of the main activities, important transitions, and the overall pattern.
Overall system flow:
Smartphone Accelerometer + Gyroscope
                ↓
        Browser DeviceMotion API
                ↓
             Flask API
                ↓
        128-Sample Sensor Buffer
                ↓
          30 Feature Extraction
                ↓
        Random Forest Classifier
                ↓
        Activity Prediction
                ↓
         Activity Sequence
                ↓
            Ollama / LLM
                ↓
      Natural-Language Interpretation
                ↓
          Web Dashboard
IV. Simulation Results & Discussions
After training, the Random Forest model was successfully saved and loaded by the Flask application. The model expects 30 input features and recognizes the following six classes: LAYING, SITTING, STANDING, WALKING, WALKING_DOWNSTAIRS, and WALKING_UPSTAIRS.
The live testing part of the project was also successfully connected. When the sensor page was opened on a smartphone, a Start Sensors button was used to begin collecting motion data. The page displayed the current accelerometer and gyroscope values, the number of motion events received, and the latest sensor data sent to the server.
The Flask backend successfully received individual sensor samples and collected them until the buffer reached 128 samples. Once the buffer was full, the application calculated the 30 features and passed them to the Random Forest model. The terminal output confirmed that live predictions were being generated, for example SITTING and WALKING_DOWNSTAIRS.
The activity-sequence part of the system also worked as intended. When the prediction changed, the new activity was added to the sequence. This sequence was then displayed on the web page rather than showing only one isolated prediction.
The LLM component was successfully tested after enough activity changes had been collected. For one of the generated sequences, the LLM identified activities such as WALKING_DOWNSTAIRS, WALKING, WALKING_UPSTAIRS, LAYING, and SITTING. It also described the transitions between these activities and gave an overall interpretation of the sequence.
During real-world testing, I also noticed that the predictions were not always identical to the activity I was actually performing. For example, while walking, the system sometimes predicted WALKING_DOWNSTAIRS. When the phone was kept on my lap or remained almost stationary, it sometimes predicted SITTING or LAYING. This is an important observation because the model is working with the motion and orientation of the phone rather than directly observing the person's body.
These results show that the complete pipeline is functional, but they also show the limitations of using smartphone sensors in an uncontrolled environment. Phone placement, orientation, sampling behaviour, and the difference between the training data and live sensor data can all affect the final prediction. Improving these aspects would be an important direction for future work.
Overall, the project successfully demonstrated the complete process from collecting live sensor measurements to generating a machine-learning prediction and then using an LLM to explain a sequence of those predictions.
V. Conclusions
In this internship, I developed a complete Human Activity Recognition prototype using smartphone accelerometer and gyroscope data. The project helped me understand the complete workflow involved in a machine-learning application, starting from data processing and feature extraction and continuing through model training, deployment, and live testing.
A Random Forest classifier was trained using 30 statistical features extracted from six sensor signals. The trained model was then integrated into a Flask application, which receives live smartphone sensor readings and performs activity prediction using 128-sample windows.
The project was further extended by maintaining a sequence of changing activity predictions and passing this sequence to an LLM through Ollama. The LLM was able to convert the sequence into a natural-language explanation of the activities and their transitions. This provided an additional layer of interpretation on top of the machine-learning predictions.
The live experiments also made it clear that a model trained on a fixed dataset may behave differently when it is used with a real smartphone. In particular, phone placement and orientation had a noticeable effect on the predictions. Therefore, future improvements could include collecting more live smartphone data, calibrating the sensor readings, using better sampling and windowing, applying prediction smoothing, and retraining the model with data collected from different phone placements.
Overall, the internship provided practical experience in machine learning, sensor data processing, Flask-based deployment, real-time data handling, and the integration of generative AI with a conventional machine-learning system.
VI. References
[1] UCI Machine Learning Repository, Human Activity Recognition Using Smartphones Dataset.
https://cdn.uci-ics-mlr-prod.aws.uci.edu/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones/UCI+HAR%20Dataset.zip
[2] Scikit-learn documentation, RandomForestClassifier.
[3] Flask documentation, Python web framework.
[4] Ollama documentation, local runtime for large language models.
[5] NumPy documentation, numerical computing in Python.
[6] Pandas documentation, data analysis and manipulation in Python.
