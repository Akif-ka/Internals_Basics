import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df1 = pd.read_csv("data/training_data.csv")
df2 = pd.read_csv("data/new_data.csv")

combined = pd.concat([df1, df2])

X = combined.drop("fuel_cost", axis=1)
y = combined["fuel_cost"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with open("models/best_model.pkl", "rb") as f:
    champion = pickle.load(f)

champ_mae = mean_absolute_error(y_test, champion.predict(X_test))

new_model = type(champion)()
new_model.fit(X_train, y_train)

new_mae = mean_absolute_error(y_test, new_model.predict(X_test))

improvement = champ_mae - new_mae

action = "promoted" if improvement >= 1.0 else "kept_champion"

output = {
    "original_data_rows": len(df1),
    "new_data_rows": len(df2),
    "combined_data_rows": len(combined),
    "champion_mae": champ_mae,
    "retrained_mae": new_mae,
    "improvement": improvement,
    "min_improvement_threshold": 1.0,
    "action": action,
    "comparison_metric": "mae"
}

with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 4 DONE")
