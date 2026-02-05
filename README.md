A lightweight, production-ready REST API that classifies audio samples as Human or AI-generated voice using signal-processing features and machine-learning — deployed and live.

What This Project Does:
This API takes an audio sample (Base64-encoded) and determines whether the voice is human or AI-generated, along with a confidence score.

It’s built for GUVI AI Impact Buildathon 2026

1)Voice authentication systems
2)Deepfake / AI-voice detection
3)Research & experimentation

How It Works:
1)Audio Input (Base64)
2)Preprocessing
3)Resampled to 16kHz
4)Mono channel
5)Feature Extraction
6)Silence ratio
7)Pitch variance
8)Energy jitter
9)MFCC entropy
10)ML Model
11)Logistic Regression (scikit-learn)
12)Prediction
13)HUMAN or AI
14)Confidence score

Tech Stack
Backend: FastAPI
ML: scikit-learn
Audio Processing: librosa, numpy
Deployment: Render
Server: Uvicorn
Language: Python 3.12

Live API
Base URL: https://voice-detector-api-kcxk.onrender.com
x-api-key: 33566c9e9244521e22e4ab905491968c8a3d8d266a68b223d2220284d60f1946

Request Body
{
  "audio_base64": "BASE64_ENCODED_AUDIO"
}
Response
{
  "classification": "HUMAN",
  "confidence": 0.87,
  "language": "en",
  "model_version": "v0.1"
}
Supported Audio
1)WAV (recommended)
2)MP3 / AAC (converted internally)
3)Mono or stereo (auto-handled)

Model Training Pipeline
python -m scripts.extract_features
python -m scripts.train_model


Model is saved as:

voice_detector.pkl

Project Structure
voice-detector-api/
├── app/
│   ├── main.py
│   ├── api/
│   │   └── routes.py
│   └── utils/
│       └── vad.py
├── scripts/
│   ├── extract_features.py
│   ├── train_model.py
│   └── test_model.py
├── requirements.txt
├── render.yaml
└── README.md

Author
Harshit Sachdeva
