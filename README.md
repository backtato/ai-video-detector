# AI-Video Detector (stable 1.2.x)
FastAPI backend per analisi euristica di video/audio con output UI-friendly.

## Endpoints
- POST /analyze — upload file
- POST /analyze-url — analizza URL (richiede USE_YTDLP=1 per social)
- GET /healthz — health lightweight
- GET /readyz — diagnostica semplice
- GET|POST /cors-test — verifica CORS

Soglie conservative (real ≤ 0.35, ai ≥ 0.72). Timeout duri per ffprobe/ffmpeg/yt-dlp.
Output: result{label, ai_score, confidence, reason}, timeline_binned, peaks, hints.
