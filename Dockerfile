FROM python:3.11-slim

# --- Sistema + ffmpeg/ffprobe (necessario per ffprobe in runtime) ---
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# --- Working dir ---
WORKDIR /app

# --- Dipendenze Python ---
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- Codice ---
COPY . /app

# --- Gunicorn/port ---
ENV WEB_CONCURRENCY=2
ENV WORKER_TIMEOUT=120
ENV PORT=8000

CMD ["gunicorn", "-c", "gunicorn_conf.py", "api:app"]