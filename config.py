import os

# ----- General -----
APP_NAME = "AI-Video"
APP_VERSION = os.getenv("APP_VERSION", "0.5.0")
ENV = os.getenv("ENV", "production")

# Bytes
MB = 1024 * 1024

# Limits (aligned to project memory)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 80 * MB))           # 80 MB
RESOLVER_MAX_BYTES = int(os.getenv("RESOLVER_MAX_BYTES", 120 * MB))      # 120 MB
DOWNLOAD_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", 25))                # seconds

# Analysis thresholds
MIN_DURATION_SEC = float(os.getenv("MIN_DURATION_SEC", 4.0))
MIN_FRAMES_FOR_CONFIDENCE = int(os.getenv("MIN_FRAMES_FOR_CONFIDENCE", 64))

# Decision thresholds (tunable)
THRESH_AI = float(os.getenv("THRESH_AI", 0.66))
THRESH_ORIGINAL = float(os.getenv("THRESH_ORIGINAL", 0.34))

# Ensemble weights (sum ~= 1)
ENSEMBLE_WEIGHTS = {
    "metadata": float(os.getenv("W_METADATA", 0.33)),
    "frame_artifacts": float(os.getenv("W_FRAME", 0.44)),
    "audio": float(os.getenv("W_AUDIO", 0.23)),
}

# Calibration toggle (if set to "1", apply isotonic-like squashing)
CALIBRATION = bool(int(os.getenv("CALIBRATION", "1")))

# User Agent for resolver
DEFAULT_UA = os.getenv(
    "RESOLVER_UA",
    "AI-Video/0.5 (+https://ai-video.org; contact: admin@ai-video.org)"
)
