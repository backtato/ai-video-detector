# Dockerfile
FROM python:3.11-slim

# Evita prompt interattivi e riduce warnings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dipendenze di sistema minime per yt-dlp/opencv + ffmpeg per eventuali merge
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Requirements prima per massimizzare layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia codice
COPY . /app

# Healthcheck (FastAPI /healthz)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz').read()" || exit 1

# Avvio con Gunicorn + UvicornWorker
CMD ["gunicorn", "api:app", "-k", "uvicorn.workers.UvicornWorker", "-c", "gunicorn_conf.py"]
