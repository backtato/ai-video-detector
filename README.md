
# AI Video Detector (CPU-only) — URL Social/YouTube o Upload (<= 50MB)

Backend FastAPI che stima se un video è **AI-generated** con euristiche leggere (CPU).  
Funziona su **Render** ed accetta: **URL social/YouTube** (via `yt-dlp`) oppure **upload di file** fino a **50MB**.

## Endpoints
- `GET /healthz` → `{"status":"ok"}`
- `POST /predict`
  - **Form fields**:
    - `url` *(opzionale)*: link YouTube/TikTok/Twitter/X/Instagram/Facebook/Reddit o link diretto al file video
    - `file` *(opzionale)*: upload multipart di un video (`mp4/mov/mkv/webm/avi`) **<= 50MB**
  - **Nota**: inviare *uno* tra `url` o `file` (non entrambi).
  - **Response**: `{ "score": 0..1, "label": "ai-generated"|"real", "details": {...} }`

## Avvio in locale
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Test rapido (upload)
```bash
curl -F "file=@/percorso/video.mp4" http://localhost:8000/predict
```

### Test rapido (youtube url)
```bash
curl -F "url=https://www.youtube.com/watch?v=XXXXXXXX" http://localhost:8000/predict
```

## Docker
```bash
docker build -t ai-video-detector .
docker run -p 8000:8000 -e PORT=8000 -e MAX_UPLOAD_MB=50 ai-video-detector
```

## Perché deploya su Render
- Server bind su `0.0.0.0:$PORT` (via Gunicorn conf)
- Nessun modello pesante → avvio rapido
- `ffmpeg` presente per `yt-dlp` e OpenCV
- Healthcheck `/healthz` configurato in `render.yaml`
- Limite input 50MB lato back-end

## Configurazione
- `MAX_UPLOAD_MB` (default 50)
- `THRESHOLD` (default 0.55)
- `TARGET_FPS` (default 6)

## Limitazioni
- Si tratta di un **baseline euristico**. Per un detector ML più accurato possiamo aggiungere un modello Torch CPU con pesi precaricati nel build.
