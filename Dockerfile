# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg exiftool tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements first (per layer cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App
COPY . /app

# Porta/healthcheck
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

# Gunicorn + Uvicorn worker
CMD ["gunicorn", "-c", "gunicorn_conf.py", "api:app"]