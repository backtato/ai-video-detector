
import os
from dataclasses import dataclass

@dataclass
class Settings:
    TARGET_FPS: int = int(os.environ.get("TARGET_FPS", "6"))       # downsample temporale
    MAX_FRAMES: int = int(os.environ.get("MAX_FRAMES", "480"))     # ~80s a 6fps
    THRESHOLD: float = float(os.environ.get("THRESHOLD", "0.55"))  # soglia etichetta
    MIN_EDGE_VAR: float = float(os.environ.get("MIN_EDGE_VAR", "15.0"))
    TMP_DIR: str = os.environ.get("TMP_DIR", "/tmp/vids")
    MAX_UPLOAD_MB: int = int(os.environ.get("MAX_UPLOAD_MB", "50"))  # limite di input (MB)
    ALLOWED_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi")

settings = Settings()
