# gunicorn_conf.py — minimal, robust, Render-friendly

import os

# Bind coerente con Docker/Render
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")

# Concurrency: istanze piccole → 1 worker
workers = int(os.getenv("WEB_CONCURRENCY", "1"))
threads = int(os.getenv("GUNICORN_THREADS", "1"))
worker_class = "uvicorn.workers.UvicornWorker"

# Startup sicuro
preload_app = False
timeout = int(os.getenv("GUNICORN_TIMEOUT", "180"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "2"))

# Riciclo per evitare memory bloat
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "200"))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "50"))

# Log
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")

# Proxy/forward headers di Render
forwarded_allow_ips = "*"