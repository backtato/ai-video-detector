# AI-Video Detector (FastAPI)

Backend FastAPI + Gunicorn per stimare la plausibilità che un video sia generato da AI.

## Endpoints

- `GET /healthz` → "ok"
- `GET /predict-get?url=...` (alias `GET /predict`)  
- `POST /predict` (form-data: `url` **oppure** `file`; opzionale `cookies_b64`)
- `POST /analyze` (alias WP)  
- `POST /analyze-url` (alias WP)

Output (JSON):
```json
{
  "ai_score": 0.51,
  "confidence": 0.72,
  "details": {
    "parts": {"metadata": 0.50, "frame_artifacts": 0.54, "audio": 0.45},
    "ffprobe": {...}
  }
}
