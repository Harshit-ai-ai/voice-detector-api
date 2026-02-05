import json
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

with open("features.json") as f:
    data = json.load(f)

X = []
y = []

for row in data:
    X.append([
        row["silence_ratio"],
        row["pitch_variance"],
        row["energy_jitter"],
        row["mfcc_entropy"]
    ])
    y.append(row["label"])

X = np.array(X)
y = np.array(y)

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "voice_detector.pkl")
print("Model saved as voice_detector.pkl")
