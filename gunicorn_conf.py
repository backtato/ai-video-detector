bind = "0.0.0.0:8000"

# Conservativo: una sola pipeline pesante per volta
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"

# Allineati ai timeouts interni (yt-dlp 180s, ffmpeg 60s, ecc.)
timeout = 240
graceful_timeout = 30
keepalive = 5

# Riciclo periodico per evitare leak/latenze accumulate in librerie native
max_requests = 200
max_requests_jitter = 20

# Log essenziali
accesslog = "-"
errorlog = "-"
loglevel = "info"
