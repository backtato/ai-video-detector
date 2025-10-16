FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# ffmpeg + librerie minime per OpenCV e yt-dlp
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY gunicorn_conf.py ./gunicorn_conf.py

EXPOSE 8000
ENV PORT=8000

CMD ["gunicorn", "-c", "gunicorn_conf.py", "app.main:app"]