from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import numpy as np
import json
import os
from datetime import datetime

app = FastAPI()

# Load model
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ FIXED VALIDATION (allows drift data)
class InputData(BaseModel):
    trip_distance_km: float = Field(..., ge=0)
    vehicle_age_years: int = Field(..., ge=0)
    load_weight_tons: float = Field(..., ge=0)
    route_type: int = Field(..., ge=1, le=3)

@app.get("/health")
def health():
    return {"alive": True, "service": "FleetTrack fuel_cost API"}

@app.post("/estimate")
def estimate(data: InputData):
    features = np.array([[ 
        data.trip_distance_km,
        data.vehicle_age_years,
        data.load_weight_tons,
        data.route_type
    ]])

    prediction = model.predict(features)[0]

    log = {
        "timestamp": str(datetime.now()),
        "input": data.dict(),
        "prediction": float(prediction)
    }

    os.makedirs("logs", exist_ok=True)
    with open("logs/predictions.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")

    return {"prediction": float(prediction)}
