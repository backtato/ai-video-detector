# Configuration and weights for the ensemble.
# Weights should be re-calibrated when plugging real detectors.

ENSEMBLE_WEIGHTS = {
    "metadata": 0.25,
    "frame_artifacts": 0.55,
    "audio": 0.20
}

# Logistic calibration parameters (Platt-like) for the raw ensemble score
CALIBRATION = {
    "a": 3.0,   # slope
    "b": -1.5   # bias
}

# Minimum quality constraints
MIN_FRAMES_FOR_CONFIDENCE = 24   # ~1 sec at 24fps
MIN_DURATION_SEC = 1.0
