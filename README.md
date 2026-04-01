# DataScience-Project
Developed a real-time human activity recognition system using accelerometer data (x, y, z) from the Phyphox mobile app. Processed and trained a machine learning model to classify activities like walking and jumping, enabling live predictions on incoming sensor data.

Real-Time Human Activity Recognition System

Overview
This project implements a real-time human activity recognition system using smartphone accelerometer data (x, y, z axes). Data is collected via the Phyphox mobile app and processed through a machine learning pipeline to classify activities such as walking and jumping.

The system supports both offline model training and real-time inference using live sensor data.

Features

Collects real-time accelerometer data (x, y, z)
Data preprocessing and segmentation pipeline
Machine learning model for activity classification
Real-time prediction from live sensor input
Visualization of sensor data and outputs



How It Works

Data Collection
Accelerometer data (x, y, z) is collected using the Phyphox app.
Preprocessing
Noise handling
Normalization
Segmentation into time windows
Feature Extraction
Statistical features are extracted from segmented data.
Model Training
A machine learning model is trained on labeled data.
Real-Time Prediction
Incoming sensor data is processed and classified in real time.

Installation

git clone https://github.com/your-username/DataScience-Project.git

cd DataScience-Project
pip install -r requirements.txt

Usage

Train the Model (Optional)
python classifier.py

Run Real-Time Prediction
python realtime_app.py

Note: A pre-trained model is already included. Training is NOT required to run the system.

Real-Time Data Setup (Phyphox)

This project uses the Phyphox mobile app to stream live accelerometer data.

Steps:

Install Phyphox on your phone
Open the app and select "Acceleration with g"
Tap the three-dot menu (top-right corner)
Enable "Remote Access"
A URL will appear like:
http://192.168.x.x:8080/

Connect Your Laptop

Ensure phone and laptop are on the same WiFi
Open the URL in your laptop browser
Start the experiment (press play)

Update the Code

Open realtime_app.py and replace:

PHYPHOX_URL = "http://10.216.130.141:8080/
"

with your own URL:

PHYPHOX_URL = "http://your-device-ip:8080/
"

Run the System

python realtime_app.py

Common Issues

Wrong IP address → connection fails
Different WiFi network → no communication
Remote access not enabled → no data
Experiment not started → empty output

Limitations

Supports limited activities (walking, jumping)
Accuracy depends on training data quality
Real-time performance depends on input stream

Future Improvements

Add more activities (running, sitting, etc.)
Improve accuracy using advanced models
Deploy as a web or mobile application
Automate device connection (remove manual IP setup)

Tech Stack

Python
NumPy
Pandas
Scikit-learn
Joblib
