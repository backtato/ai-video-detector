# gunicorn_conf.py — minimal, robust, Render-friendly

import os

# Bind: Render/Docs -> 0.0.0.0:8000 (Dockerfile EXPOSE 8000)
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")

# Concurrency: one worker is safer on small instances
workers = int(os.getenv("WEB_CONCURRENCY", "1"))
threads = int(os.getenv("GUNICORN_THREADS", "1"))
worker_class = "uvicorn.workers.UvicornWorker"

# Startup/runtime safety
preload_app = False
timeout = int(os.getenv("GUNICORN_TIMEOUT", "180"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "2"))

# Recycle to avoid memory bloat
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "200"))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "50"))

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")

# Networking hygiene
forwarded_allow_ips = "*"