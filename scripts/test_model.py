import joblib
import numpy as np
from app.utils.vad import silence_ratio, pitch_variance, energy_jitter, mfcc_entropy
import librosa
import sys

model = joblib.load("voice_detector.pkl")

def extract_features(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    return [
        silence_ratio(y, sr),
        pitch_variance(y, sr),
        energy_jitter(y),
        mfcc_entropy(y)
    ]

audio = sys.argv[1]
features = extract_features(audio)
X = np.array(extract_features(audio)).reshape(1, -1)
pred = model.predict(X)[0]
prob = model.predict_proba(X)[0]

print("Prediction:", "AI" if pred == 1 else "Human")
print("Confidence:", prob)
