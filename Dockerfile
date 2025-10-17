# CPU-only, Python 3.11 slim
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps: ffmpeg/ffprobe per yt-dlp e analisi media
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Crea dir app
WORKDIR /app

# Install requirements con caching corretto
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copia il resto del codice
COPY . /app

# Healthcheck (opzionale; Render usa /healthz)
EXPOSE 8000

# Gunicorn + UvicornWorker (config in gunicorn_conf.py)
CMD ["gunicorn", "-c", "gunicorn_conf.py", "app:app"]
