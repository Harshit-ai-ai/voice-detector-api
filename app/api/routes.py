from fastapi import APIRouter, Depends
from app.schemas.request_response import DetectRequest, DetectResponse
from app.api.auth import verify_api_key
from app.ml.features import extract_mfcc_from_base64
from app.utils.audio import decode_audio_base64
from app.utils.vad import (
    silence_ratio,
    pitch_variance,
    energy_jitter,
    mfcc_entropy
)
from app.core.heuristic_classifier import heuristic_score
import librosa
import numpy as np

router = APIRouter()

@router.post("/detect", response_model=DetectResponse)
@router.post("/detect", response_model=DetectResponse)
def detect_voice(
    request: DetectRequest,
    api_key: str = Depends(verify_api_key)
):
    y = decode_audio_base64(request.audio_base64)
    sr = 16000
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = {
        "silence_ratio": silence_ratio(y, sr),
        "pitch_variance": pitch_variance(y, sr),
        "energy_jitter": energy_jitter(y),
        "mfcc_entropy": mfcc_entropy(mfcc)
    }
    human_score = heuristic_score(features)

    classification = "HUMAN" if human_score >= 0.5 else "AI_GENERATED"
    return {
        "classification": classification,
        "confidence": round(human_score, 2),
        "language": request.language,
        "model_version": "v1.0-heuristic",
        "explainability": features
    }
