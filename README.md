FROM python:3.11-slim

# === Meta informazioni ===
LABEL maintainer="Backtato"
LABEL author="Backtato"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (valuta se rimuovere tesseract se non usato)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg exiftool curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["gunicorn", "-c", "gunicorn_conf.py", "api:app"]
