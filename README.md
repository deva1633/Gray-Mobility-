🚨 Hybrid Anomaly Detection System

📌 Project Overview

This project implements a hybrid anomaly detection system designed to detect abnormal patterns in time-series or operational data.
It combines:
Unsupervised Machine Learning (Isolation Forest)
Rule-based validation logic
Confidence scoring & alert suppression
The goal is to reduce false alerts while maintaining high sensitivity to critical anomalies.

🧠 System Architecture

Raw Data
   ↓
Feature Engineering
   ↓
ML-based Anomaly Detection (Isolation Forest)
   ↓
Rule-Based Validation Layer
   ↓
Risk Scoring & Alert Decision
   ↓
CSV Outputs (Logs, Alerts, Confidence)

✨ Key Features

Hybrid detection (ML + Rules)
False alert suppression logic
Confidence scoring for alerts
Time-window based validation
Modular design (training & inference separated)
CSV-based outputs for easy auditing
.
├── data/
│   ├── raw_data.csv
│   ├── features.csv
│   └── inference_data.csv
│
├── models/
│   └── isolation_forest.pkl
│
├── outputs/
│   ├── anomaly_scores.csv
│   ├── rule_based_output.csv
│   └── final_alerts.csv
│
├── train_anomaly_model.py
├── inference_pipeline.py
├── rule_engine.py
├── validation_runner.py
├── requirements.txt
└── README.md

🧪 Model Training
This project uses unsupervised learning, so training does not require labeled data.

Training Script
python train_anomaly_model.py

What happens during training:
Loads engineered features from features.csv
Trains an Isolation Forest model
Saves the trained model to models/
Training is designed to be offline, allowing periodic retraining without affecting inference.

⚙️ Inference & Detection

Run inference
python inference_pipeline.py

Inference steps:
Load trained model
Score incoming data
Identify anomalous samples
Generate anomaly scores

📏 Rule-Based Validation Layer

The rule engine validates ML predictions using domain-defined thresholds such as:
Risk score limits
Time-window consistency
Suppression conditions
Confidence thresholds
This layer helps:
Reduce false positives
Add explainability
Enforce business constraints

🧮 Risk Scoring & Alerts

Each record is assigned:

Field	Description
risk_score	Combined anomaly severity
risk_level	GREEN / YELLOW / RED
confidence	Reliability of detection
alert_flag	0 (no alert) or 1 (alert)
suppression_reason	Explanation if suppressed

📤 Outputs

All results are stored as CSV files in the outputs/ directory:
anomaly_scores.csv
rule_based_output.csv
final_alerts.csv
These outputs can be:
Audited
Visualized
Used for downstream systems

📦 Installation
python -m venv venv
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt

🛠 Technologies Used
Python
Pandas, NumPy
Scikit-learn
Isolation Forest
Rule-based logic
CSV-driven pipelines

🚀 Future Enhancements

Automated retraining triggers
Model versioning
Real-time streaming support
Dashboard visualization
Alert escalation logic

🎯 Use Cases

Industrial monitoring
Operational risk detection
Sensor anomaly detection
Preventive maintenance
Fraud or outlier detection

📝 Notes

This project emphasizes:
Explainability
Low false-alert rates
Production-ready structure
It is suitable for internship evaluations, academic projects, and real-world anomaly detection pipelines.
