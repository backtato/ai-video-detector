# app.py
import os
import json
import tempfile
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import (
    CORS_ALLOW_ORIGINS, MAX_UPLOAD_BYTES, RESOLVER_MAX_BYTES,
    MIN_DURATION_SEC, MIN_FRAMES_FOR_CONFIDENCE,
    ENSEMBLE_WEIGHTS, CALIBRATION, THRESH_AI, THRESH_ORIGINAL
)

from utils.resolver import resolve_to_media_file
from detectors.metadata import ffprobe, score_metadata
from detectors.frame_artifacts import score_frame_artifacts
from detectors.audio import score_audio
from calibration import calibrate, combine_scores


app = FastAPI(title="AI-Video", version="0.6")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ALLOW_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


def _analyze_local_file(path: str) -> Dict[str, Any]:
    """Analisi MVP combinando metadata + artefatti frame + audio."""
    meta = ffprobe(path)
    duration = float(meta.get("format", {}).get("duration", 0) or 0)
    # punteggio per componente
    s_meta = score_metadata(meta)
    s_frame, frame_detail = score_frame_artifacts(path, target_frames=24)
    s_audio = score_audio(path)

    raw_scores = {
        "metadata": s_meta,
        "frame_artifacts": s_frame,
        "audio": s_audio,
    }
    combined = combine_scores(raw_scores, ENSEMBLE_WEIGHTS)
    ai_score, confidence = calibrate(combined, CALIBRATION, duration, frame_detail.get("n_frames", 0))

    verdict = "AI-likely" if ai_score >= THRESH_AI else ("Original-likely" if ai_score <= THRESH_ORIGINAL else "Inconclusive")

    return {
        "result": verdict,
        "ai_score": round(ai_score, 2),
        "confidence": round(confidence, 2),
        "details": {
            "metadata": {"score": round(s_meta, 2), "ffprobe": meta.get("format", {})},
            "frame_artifacts": {"score": round(s_frame, 2), **frame_detail},
            "audio": {"score": round(s_audio, 2)},
            "duration_sec": duration,
        },
    }


@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    # size guard: FastAPI non fornisce size di default; fidiamoci del reverse proxy/Render
    suffix = os.path.splitext(file.filename or "")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File too large")
        tmp.write(data)
        tmp.flush()
        path = tmp.name

    try:
        report = _analyze_local_file(path)
        return JSONResponse(report)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


@app.post("/analyze-url")
async def analyze_url(
    request: Request,
    url_form: Optional[str] = Form(default=None),
    payload: Optional[Dict[str, Any]] = Body(default=None),
):
    # Supporta sia form-data/urlencoded (campo "url") sia JSON { "url": "..." }
    url = url_form
    if not url and payload and isinstance(payload, dict):
        url = payload.get("url")

    if not url or not isinstance(url, str):
        raise HTTPException(status_code=400, detail="Missing 'url'")

    # Risolve l'URL a un vero media file (scarica ~primi 20s) con limiti e timeout
    path = await resolve_to_media_file(url, max_bytes=RESOLVER_MAX_BYTES)
    if not path:
        raise HTTPException(status_code=422, detail="Unable to resolve media from URL")

    try:
        report = _analyze_local_file(path)
        return JSONResponse(report)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
