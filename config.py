import os

# ====== Thresholds & calibration ======
ENSEMBLE_WEIGHTS = {
    "metadata": 0.35,
    "frame_artifacts": 0.45,
    "audio": 0.20,   # NB: feature semplice; valuta di abbassarla se “piatta”
}
CALIBRATION = {
    "logit_a": 1.0,
    "logit_b": 0.0,
}

# ====== Confidence policy ======
MIN_FRAMES_FOR_CONFIDENCE = 60
MIN_DURATION_SEC = 4.0

# Durata-consapevole: offset e pendenza più prudenti su clip brevi
CONFIDENCE_BASE = 0.2         # prima era 0.3
CONFIDENCE_SCALE = 0.8        # prima era 0.7
CONFIDENCE_MAX_FRAMES = 180   # frames per arrivare a conf. piena
CONFIDENCE_MAX_DURATION = 120 # sec per arrivare a conf. piena

THRESH_AI = 0.60
THRESH_ORIGINAL = 0.40

# ====== Resolver / networking ======
# Se vuoto: consenti http/https pubblici (blocca IP privati/localhost)
RESOLVER_ALLOWLIST = os.getenv("RESOLVER_ALLOWLIST", "")
# Cap byte per download da URL (413 se oltre)
RESOLVER_MAX_BYTES = int(os.getenv("RESOLVER_MAX_BYTES", str(120 * 1024 * 1024)))  # 120MB
# Cap byte per upload locale (413 se oltre)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(80 * 1024 * 1024)))       # 80MB

# ffmpeg HLS options
FFMPEG_USER_AGENT = os.getenv("FFMPEG_USER_AGENT", "Mozilla/5.0 (AI-Video/1.0)")
FFMPEG_RW_TIMEOUT_US = int(os.getenv("FFMPEG_RW_TIMEOUT_US", "15000000"))  # 15s
HLS_SAMPLE_SECONDS = int(os.getenv("HLS_SAMPLE_SECONDS", "8"))

# ====== Sampling policy ======
# su video lunghi, limitiamo il numero di frame estratti
MAX_SAMPLED_FRAMES = int(os.getenv("MAX_SAMPLED_FRAMES", "180"))

# ====== YouTube / social ======
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
