FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend /app/backend

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000
CMD ["gunicorn", "-c", "backend/gunicorn_conf.py", "backend.app:app"]
