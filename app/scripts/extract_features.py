import os
import json
import librosa
import numpy as np
from app.utils.vad import silence_ratio, pitch_variance, energy_jitter, mfcc_entropy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SR = 16000

def extract_features(audio_path):
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13)

    return {
        "silence_ratio": silence_ratio(y, SR),
        "pitch_variance": pitch_variance(y, SR),
        "energy_jitter": energy_jitter(y),
        "mfcc_entropy": mfcc_entropy(mfcc)
    }


def process_folder(folder, label):
    data = []
    for file in os.listdir(folder):
        if file.endswith(".wav") or file.endswith(".mp3"):
            path = os.path.join(folder, file)
            features = extract_features(path)
            features["label"] = label
            data.append(features)
    return data


if __name__ == "__main__":
    dataset = []
    dataset += process_folder("data/human", 1)
    dataset += process_folder("data/ai", 0)

    with open("features.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved {len(dataset)} samples")
