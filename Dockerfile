FROM python:3.11-slim

# Evita bytecode e output bufferizzato
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dipendenze di sistema (ffmpeg include ffprobe)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installa deps Python con cache layer-friendly
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il codice
COPY . /app

# Porta dinamica di Render
ENV PORT=8000

EXPOSE 8000
# Avvio robusto con UvicornWorker e timeout ragionevoli
CMD gunicorn -k uvicorn.workers.UvicornWorker app:app \
    --bind 0.0.0.0:${PORT} \
    --timeout 120 \
    --graceful-timeout 30 \
    --workers 1 \
    --threads 4
