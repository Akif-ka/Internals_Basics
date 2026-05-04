# 🚀 MLOps Lab CIE Project

## 📌 Project Title
FleetTrack Fuel Cost Prediction System

---

## 📖 Overview
This project implements an end-to-end MLOps pipeline for predicting fuel cost using machine learning.  
It includes model training, API deployment, traffic simulation, drift detection, and retraining.

---

## 🛠️ Technologies Used
- Python
- Scikit-learn
- FastAPI
- MLflow
- Pandas, NumPy

---

## 📂 Project Structure
MLOPs_Lab_CIE/
│
├── data/ # Training and new datasets
├── src/ # Source code files
│ ├── train.py
│ ├── api.py
│ ├── simulate_traffic.py
│ ├── monitor.py
│ ├── retrain.py
│
├── models/ # Saved ML model
├── logs/ # Prediction logs
├── results/ # Output JSON files
│ ├── step1_s1.json
│ ├── step3_s5.json
│ ├── step4_s8.json
│
├── requirements.txt
└── README.md

---

## ⚙️ Steps Performed

### 🔹 Task 1: Model Training
- Trained Linear Regression and Random Forest models
- Logged experiments using MLflow
- Selected best model based on MAE

---

### 🔹 Task 2: API Deployment
- Built REST API using FastAPI
- Endpoint `/estimate` predicts fuel cost
- Logs predictions into `logs/predictions.jsonl`

---

### 🔹 Task 3: Drift Detection
- Simulated traffic using synthetic data
- Compared live vs training distribution
- Detected drift based on threshold

---

### 🔹 Task 4: Retraining
- Combined old + new data
- Retrained model
- Promoted model if performance improved

---

## ▶️ How to Run

```bash
# Train model
python src/train.py

# Start API
uvicorn src.api:app --reload

# Simulate traffic
python src/simulate_traffic.py

# Monitor drift
python src/monitor.py

# Retrain model
python src/retrain.py

Results
Drift detected successfully
Model improved after retraining
Final model promoted

Conclusion

This project demonstrates a complete MLOps workflow including training, deployment, monitoring, and retraining.
