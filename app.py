import os
import tempfile
import shutil
from typing import Dict

from fastapi import FastAPI, UploadFile, File, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import (
    APP_NAME, APP_VERSION, ENSEMBLE_WEIGHTS, CALIBRATION,
    MIN_FRAMES_FOR_CONFIDENCE, MIN_DURATION_SEC,
    THRESH_AI, THRESH_ORIGINAL, MAX_UPLOAD_BYTES
)
from .calibration import combine_scores
from .detectors.metadata import ffprobe, score_metadata
from .detectors.frame_artifacts import score_frame_artifacts
from .detectors.audio import score_audio
from .utils.resolver import fetch_to_temp
from .models.schema import AnalyzeResult, URLPayload

app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def _classify(ai_score: float) -> str:
    if ai_score >= THRESH_AI: return "Probabile AI"
    if ai_score <= THRESH_ORIGINAL: return "Probabile autentico"
    return "Inconcludente"

def _analyze_file(tmp_path: str) -> AnalyzeResult:
    # 1) metadata
    meta = ffprobe(tmp_path)
    m = score_metadata(meta)

    # 2) frames
    f = score_frame_artifacts(tmp_path)

    # 3) audio (from metadata for robustness)
    a = score_audio(meta)

    parts = {
        "metadata": m["score"],
        "frame_artifacts": f["score"],
        "audio": a["score"],
    }
    ai_score = combine_scores(parts, ENSEMBLE_WEIGHTS, calibrate=CALIBRATION)
    verdict = _classify(ai_score)

    # crude confidence proxy: number of frames used & duration if available
    frames_used = f["details"].get("frames_used", 0)
    confidence = min(1.0, max(0.3, frames_used / max(1, MIN_FRAMES_FOR_CONFIDENCE)))

    details: Dict = {
        "metadata": m["details"],
        "frame_artifacts": f["details"],
        "audio": a["details"],
        "weights": ENSEMBLE_WEIGHTS,
    }

    return AnalyzeResult(
        verdict=verdict,
        ai_score=round(float(ai_score), 4),
        confidence=round(float(confidence), 2),
        parts=parts,
        details=details,
    )

@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION}

@app.post("/analyze", response_model=AnalyzeResult)
async def analyze(file: UploadFile = File(...)):
    if file.size and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (> {MAX_UPLOAD_BYTES} bytes).")
    suffix = os.path.splitext(file.filename or "")[-1] or ".bin"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                if f.tell() > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=f"File too large (> {MAX_UPLOAD_BYTES} bytes).")
        res = _analyze_file(tmp)
        return JSONResponse(res.dict())
    finally:
        try: os.remove(tmp)
        except Exception: pass

@app.post("/analyze-url", response_model=AnalyzeResult)
def analyze_url(payload: URLPayload = Body(...)):
    if not payload.url or not payload.url.strip():
        raise HTTPException(status_code=400, detail="Missing url")
    tmp = fetch_to_temp(payload.url.strip())
    try:
        res = _analyze_file(tmp)
        # add resolver note
        res.details["resolver"] = {"downloaded_bytes": os.path.getsize(tmp)}
        return JSONResponse(res.dict())
    finally:
        try: os.remove(tmp)
        except Exception: pass
