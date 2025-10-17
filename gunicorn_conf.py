import multiprocessing
import os

bind = "0.0.0.0:8000"

# Worker Uvicorn per ASGI
worker_class = "uvicorn.workers.UvicornWorker"

# Workers/threads conservative per CPU-only
workers = int(os.getenv("WEB_CONCURRENCY", str(max(2, multiprocessing.cpu_count() // 2))))
threads = int(os.getenv("WEB_THREADS", "2"))

# Download/analisi possono durare: alza timeout
timeout = int(os.getenv("GUNICORN_TIMEOUT", "180"))
keepalive = 5

# Log base
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOGLEVEL", "info")
