import base64
import io
import librosa
import numpy as np


def decode_audio_base64(audio_base64: str, sr: int = 16000):
    """
    Decodes a Base64 MP3 string into a waveform numpy array
    """
    try:
        audio_bytes = base64.b64decode(audio_base64)
        audio_buffer = io.BytesIO(audio_bytes)

        y, _ = librosa.load(audio_buffer, sr=sr, mono=True)
        return y

    except Exception as e:
        raise ValueError(f"Audio decoding failed: {str(e)}")
