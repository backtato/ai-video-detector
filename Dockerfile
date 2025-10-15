FROM python:3.11-slim

# Dipendenze di sistema minime (ffmpeg per HLS e probing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Porta di default usata dall'app
EXPOSE 8000

# Avvio robusto in prod
CMD ["gunicorn","-k","uvicorn.workers.UvicornWorker","app:app","--bind","0.0.0.0:8000","--workers","2","--threads","4","--timeout","180","--graceful-timeout","30"]
