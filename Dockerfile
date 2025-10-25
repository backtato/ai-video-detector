FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg exiftool tesseract-ocr curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements first
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App
COPY . /app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["gunicorn", "-c", "gunicorn_conf.py", "api:app"]
