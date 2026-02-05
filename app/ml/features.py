import base64
import io
import numpy as np
import librosa
import soundfile as sf


def extract_mfcc_from_base64(
    audio_base64: str,
    sample_rate: int = 16000,
    n_mfcc: int = 13
):
    # 1️⃣ Decode base64 → bytes
    audio_bytes = base64.b64decode(audio_base64)

    # 2️⃣ Bytes → waveform
    with io.BytesIO(audio_bytes) as f:
        waveform, sr = sf.read(f)

    # 3️⃣ Convert to mono if stereo
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    # 4️⃣ Resample (important!)
    if sr != sample_rate:
        waveform = librosa.resample(
            waveform, orig_sr=sr, target_sr=sample_rate
        )

    # 5️⃣ Extract MFCCs
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=n_mfcc
    )

    # Shape: (n_mfcc, time)
    return mfcc
