import multiprocessing
import os

bind = "0.0.0.0:8000"
worker_class = "uvicorn.workers.UvicornWorker"

workers = int(os.getenv("WEB_CONCURRENCY", str(max(2, multiprocessing.cpu_count() // 2))))
threads = int(os.getenv("WEB_THREADS", "2"))

timeout = int(os.getenv("GUNICORN_TIMEOUT", "180"))
keepalive = 5

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOGLEVEL", "info")

# 👇 Indicazione esplicita del modulo e dell'oggetto FastAPI
wsgi_app = "api:app"
