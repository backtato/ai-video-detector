import os, random
bind = "0.0.0.0:8000"
workers = int(os.getenv("WEB_CONCURRENCY", "1"))
worker_class = "uvicorn.workers.UvicornWorker"
preload_app = False
timeout = int(os.getenv("GUNICORN_TIMEOUT", "180"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL", "30"))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "2"))
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "200"))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", str(random.randint(30,80))))
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOGLEVEL", "info")
