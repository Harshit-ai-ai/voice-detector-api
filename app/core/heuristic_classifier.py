def heuristic_score(features):
    score = 0.0

    if features["silence_ratio"] < 0.15:
        score += 0.2
    if features["pitch_variance"] > 20:
        score += 0.2
    if features["energy_jitter"] > 0.01:
        score += 0.2
    if features["mfcc_entropy"] > 2.5:
        score += 0.2

    return min(score, 1.0)
