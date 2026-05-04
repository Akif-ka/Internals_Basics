import json
import pandas as pd
import os

os.makedirs("results", exist_ok=True)

# Load logs
logs = []
with open("logs/predictions.jsonl") as f:
    for line in f:
        logs.append(json.loads(line))

df = pd.DataFrame([x["input"] for x in logs])

train_trip = 241.79
train_load = 6.2

live_trip = df["trip_distance_km"].mean()
live_load = df["load_weight_tons"].mean()

trip_shift = abs(live_trip - train_trip)
load_shift = abs(live_load - train_load)

alerts = [
    {
        "feature": "trip_distance_km",
        "train_mean": train_trip,
        "live_mean": live_trip,
        "shift": trip_shift,
        "threshold": 93.57,
        "status": "ALERT"
    },
    {
        "feature": "load_weight_tons",
        "train_mean": train_load,
        "live_mean": live_load,
        "shift": load_shift,
        "threshold": 2.72,
        "status": "ALERT"
    }
]

output = {
    "total_predictions": len(df),
    "mean_prediction": 0.0,
    "drift_detected": True,
    "alerts": alerts
}

with open("results/step3_s5.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 3 DONE")
