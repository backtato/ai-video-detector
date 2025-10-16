
# AI Video Detector (CPU-only) — URL Social/YouTube o Upload (<= 50MB)

Backend FastAPI che stima se un video è **AI-generated** con euristiche leggere (CPU).
Funziona su **Render** ed accetta: **URL social/YouTube** (via `yt-dlp`) oppure **upload di file** fino a **50MB**.

## Endpoints
- `GET /` → form HTML minimale per test
- `GET /healthz` → `{"status":"ok"}`
- `POST /predict` → form-data con `url` *oppure* `file`
- `GET /predict-get?url=...` → comodo da browser/link

## Esempi
```bash
# Upload file
curl -F "file=@/percorso/video.mp4" http://localhost:8000/predict

# Link YouTube/social
curl -F "url=https://www.youtube.com/watch?v=XXXXXXXX" http://localhost:8000/predict

# GET rapido da browser
open "http://localhost:8000/predict-get?url=https://www.youtube.com/watch?v=XXXXXXXX"
```

## Docker
```bash
docker build -t ai-video-detector .
docker run -p 8000:8000 -e PORT=8000 -e MAX_UPLOAD_MB=50 ai-video-detector
```
