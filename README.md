# AI Video Plausibility Detector (MVP)

**Goal:** Provide a probability that a given video is **AI-generated** vs **camera-original**, with transparent uncertainty.
This is a scaffolding for a real platform: plug in stronger models as they become available.

## Features
- **FastAPI** backend: `POST /analyze` accepts a video file.
- **Heuristic ensemble (placeholder)**:
  - Metadata inspection (container/codec anomalies).
  - Frame-level artifact scores (noise consistency, high-frequency energy, block artifacts).
  - (Optional) Audio continuity check.
- **Probability calibration** via logistic squashing with uncertainty guardrails.
- **Explainability**: returns per-detector scores and a combined plausibility with confidence bands.
- **Ready for production**: includes Dockerfile and requirements.

> ⚠️ This MVP is **not** a reliable detector on its own. Replace the heuristic detectors with trained models
(e.g., deepfake detectors, temporal transformers, watermark scanners, signature verifiers) before real-world use.
Always present outputs as probabilistic with confidence and disclaimers.

## Quickstart

### 1) Local (Python)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then call:
```bash
curl -X POST "http://localhost:8000/analyze"   -F "file=@/path/to/video.mp4"
```

### 2) Docker
```bash
docker build -t ai-video-detector:latest .
docker run --rm -p 8000:8000 ai-video-detector:latest
```

## API
`POST /analyze` multipart form with `file`.
**Response**
```json
{
  "ai_plausibility": 0.73,
  "confidence": 0.62,
  "explanations": {
    "metadata": {...},
    "frame_artifacts": {...},
    "audio": {...}
  },
  "notes": ["MVP; do not rely without human review."]
}
```

## Where to plug real models
- Add your detectors under `detectors/` and update `ENSEMBLE_WEIGHTS` in `config.py`.
- Examples: watermark checkers, proprietary SDKs, CNN/ViT frame encoders + temporal models, audio generative signatures, PRNU-based camera noise checks, etc.
- Calibrate with **isotonic regression** or **Platt scaling** on a labeled validation set. See `calibration.py`.

## Legal/Ethical
- Disclose false-positive/negative risks.
- Respect terms of service and privacy laws (GDPR). Avoid storing videos unless consented.
- Provide an appeal path and never present outputs as certainties.

## Roadmap
- Swap heuristics with trained models.
- Add async task queue and object storage.
- Add web dashboard (Next.js) and user auth.
- Add dataset evaluation harness and model cards.
