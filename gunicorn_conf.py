import os

bind = "0.0.0.0:8000"
workers = int(os.getenv("WEB_CONCURRENCY", "2"))
worker_class = "uvicorn.workers.UvicornWorker"
timeout = int(os.getenv("WORKER_TIMEOUT", "120"))
graceful_timeout = 30
keepalive = 5