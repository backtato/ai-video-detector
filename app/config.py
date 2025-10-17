# config.py
from typing import Optional
import os

# Limiti & timeouts
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 50 * 1024 * 1024))   # 50 MB
RESOLVER_MAX_BYTES = int(os.getenv("RESOLVER_MAX_BYTES", 80 * 1024 * 1024))
DOWNLOAD_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", 45))                  # sec
FFMPEG_TRIM_SECONDS = int(os.getenv("FFMPEG_TRIM_SECONDS", 20))            # primi 20s

# CORS
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")

# Soglie/parametri (MVP dimostrativo)
ENSEMBLE_WEIGHTS = {"metadata": 0.34, "frame_artifacts": 0.33, "audio": 0.33}
CALIBRATION = {"bias": 0.0, "scale": 1.0}
MIN_FRAMES_FOR_CONFIDENCE = 12
MIN_DURATION_SEC = 3.0
THRESH_AI = 0.65
THRESH_ORIGINAL = 0.35

# Path tmp
TMP_DIR = os.getenv("TMP_DIR", "/tmp")
