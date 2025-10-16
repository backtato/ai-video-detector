FROM python:3.11-slim

# Dipendenze di sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Reqs dalla root del repo
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copia tutto il codice (root → /app)
COPY . /app

# Env
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

# Avvio (app.py è in root e l'istanza FastAPI si chiama "app")
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "1", "-b", "0.0.0.0:8000", "app:app"]
