# gunicorn_conf.py
bind = "0.0.0.0:8000"
workers = 2  # fisso per ridurre RAM in parallelo
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 180
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = "info"
