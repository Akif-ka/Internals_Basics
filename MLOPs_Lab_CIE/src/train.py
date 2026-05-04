import pandas as pd
import mlflow
import mlflow.sklearn
import json
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("fuel_cost", axis=1)
y = df["fuel_cost"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("fleettrack-fuel-cost")

results = []

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

best_model = None
best_mae = float("inf")
best_name = ""

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        mlflow.log_param("model", name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.set_tag("domain", "fleet_management")

        mlflow.sklearn.log_model(model, name)

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse
        })

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_name = name

# Save model
os.makedirs("models", exist_ok=True)
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save JSON
os.makedirs("results", exist_ok=True)
output = {
    "experiment_name": "fleettrack-fuel-cost",
    "models": results,
    "best_model": best_name,
    "best_metric_name": "mae",
    "best_metric_value": best_mae
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 DONE")
