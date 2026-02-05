import numpy as np
import librosa


def silence_ratio(y, sr, threshold_db=-40):
    """
    Ratio of silent frames to total frames
    """
    rms = librosa.feature.rms(y=y)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    silent_frames = rms_db < threshold_db
    return float(np.sum(silent_frames) / len(rms_db))


def pitch_variance(y, sr):
    """
    Variance of pitch (F0)
    """
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )

    f0 = f0[~np.isnan(f0)]
    if len(f0) < 10:
        return 0.0

    return float(np.std(f0))


def energy_jitter(y):
    """
    Short-term energy variation
    """
    rms = librosa.feature.rms(y=y)[0]
    return float(np.std(rms))


def mfcc_entropy(y):
    mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13)
    mfcc_abs = np.mean(np.abs(mfcc), axis=1)  # → 1D
    mfcc_norm = mfcc_abs / (np.sum(mfcc_abs) + 1e-10)
    return -np.sum(mfcc_norm * np.log(mfcc_norm + 1e-10))
