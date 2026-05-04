import requests
import random

url = "http://127.0.0.1:8000/estimate"

# Normal data
for _ in range(35):
    data = {
        "trip_distance_km": random.uniform(10, 500),
        "vehicle_age_years": random.randint(1, 15),
        "load_weight_tons": random.uniform(0.5, 10),
        "route_type": random.randint(1, 3)
    }
    r = requests.post(url, json=data)
    print("Normal:", r.status_code)

# Drifted data
for _ in range(15):
    data = {
        "trip_distance_km": random.uniform(600, 1000),
        "vehicle_age_years": random.randint(1, 15),
        "load_weight_tons": random.uniform(10, 20),
        "route_type": random.randint(1, 3)
    }
    r = requests.post(url, json=data)
    print("Drift:", r.status_code)

print("DONE")
