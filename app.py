import os
import re
import tempfile
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import (
    ENSEMBLE_WEIGHTS,
    CALIBRATION,
    MIN_FRAMES_FOR_CONFIDENCE,
    MIN_DURATION_SEC,
    CONFIDENCE_BASE,
    CONFIDENCE_SCALE,
    CONFIDENCE_MAX_FRAMES,
    CONFIDENCE_MAX_DURATION,
    THRESH_AI,
    THRESH_ORIGINAL,
    MAX_UPLOAD_BYTES,
)

from calibration import calibrate, combine_scores
from detectors.metadata import ffprobe, score_metadata
from detectors.frame_artifacts import score_frame_artifacts
from detectors.audio import score_audio  # placeholder feature
from resolver import download_to_temp, sample_hls_to_file
from utils import ffprobe_json, video_duration_fps

app = FastAPI(title="AI-Video Detector (AI-Video)")
# === CORS (adatta ai tuoi domini frontend) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

def _confidence(frames_analyzed: int, duration_sec: float) -> float:
    f = min(1.0, frames_analyzed / max(1, CONFIDENCE_MAX_FRAMES))
    d = min(1.0, duration_sec / max(1.0, CONFIDENCE_MAX_DURATION))
    # combina prudenzialmente (media geometrica leggera)
    import math
    mix = math.sqrt(max(1e-9, f * d))
    return max(0.0, min(1.0, CONFIDENCE_BASE + CONFIDENCE_SCALE * mix))

def _sampler_target_frames(duration: float) -> int:
    # massimo frame analizzati sul totale (ad es. 180)
    from config import MAX_SAMPLED_FRAMES
    if duration <= 30:
        return min(MAX_SAMPLED_FRAMES, 120)
    if duration <= 60:
        return min(MAX_SAMPLED_FRAMES, 150)
    return MAX_SAMPLED_FRAMES

def _analyze_local_file(path: str) -> dict:
    meta = ffprobe_json(path)
    duration, fps, nb = video_duration_fps(meta)

    # === sampling adattivo ===
    target = _sampler_target_frames(duration)
    # (riusa la tua implementazione esistente di estrazione frame a passo uniforme)
    # score_frame_artifacts si occuperà di leggere i frame selezionati

    s_meta = score_metadata(meta)
    s_frames = score_frame_artifacts(path, target_frames=target)
    s_audio = score_audio(path)

    raw = {"metadata": s_meta, "frame_artifacts": s_frames, "audio": s_audio}
    combined = combine_scores(raw, ENSEMBLE_WEIGHTS)
    calibrated = calibrate(combined, CALIBRATION)

    label = "Inconclusive"
    if calibrated >= THRESH_AI:
        label = "Likely AI"
    elif calibrated <= THRESH_ORIGINAL:
        label = "Likely Original"

    conf = _confidence(s_frames.get("frames_analyzed", 0), duration)
    # Penalità su clip molto brevi
    if s_frames.get("frames_analyzed", 0) < MIN_FRAMES_FOR_CONFIDENCE or duration < MIN_DURATION_SEC:
        conf *= 0.6

    return {
        "ai_plausibility": round(calibrated, 2),
        "confidence": round(conf, 2),
        "label": label,
        "details": {
            "metadata": s_meta,
            "frame_artifacts": s_frames,
            "audio": s_audio,
            "video": {"duration": duration, "fps": fps, "frames": nb}
        }
    }

# ========== Endpoints ==========

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Upload streaming con cap (413 se troppo grande)
    suffix = os.path.splitext(file.filename or "")[-1] or ".mp4"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            total = 0
            while True:
                chunk = await file.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    tmp.close()
                    os.unlink(tmp.name)
                    return JSONResponse({"error": f"File too large (max {MAX_UPLOAD_BYTES//(1024*1024)} MB)"}, status_code=413)
                tmp.write(chunk)
            local_path = tmp.name
    except Exception as e:
        return JSONResponse({"error": f"Upload failed: {str(e)}"}, status_code=400)

    try:
        result = _analyze_local_file(local_path)
        return result
    except RuntimeError as e:
        return JSONResponse({"error": f"Processing error: {str(e)}"}, status_code=422)
    except Exception as e:
        return JSONResponse({"error": "Internal error"}, status_code=500)
    finally:
        try: os.unlink(local_path)
        except Exception: pass

@app.post("/analyze-url")
def analyze_url(url: str = Body(..., embed=True)):
    # MP4/WebM diretti → download con cap; HLS → sample 8s
    url = url.strip()
    try:
        if re.search(r"\.m3u8(\?|$)", url, re.I):
            # campiona pochi secondi per stimare
            path = sample_hls_to_file(url)
            try:
                return _analyze_local_file(path)
            finally:
                try: os.unlink(path)
                except Exception: pass
        else:
            from config import RESOLVER_MAX_BYTES
            from resolver import download_to_temp
            with download_to_temp(url, RESOLVER_MAX_BYTES) as path:
                return _analyze_local_file(path)
    except PermissionError:
        return JSONResponse({"error": "URL not allowed (policy)"}, status_code=403)
    except ValueError as e:
        if "MAX_BYTES_EXCEEDED" in str(e):
            return JSONResponse({"error": "Remote file too large"}, status_code=413)
        return JSONResponse({"error": "Invalid media"}, status_code=422)
    except Exception as e:
        # timeouts, ffmpeg error, ecc.
        return JSONResponse({"error": f"Fetch/process failed: {str(e)[-200:]}"},
                            status_code=502)

# NB: /analyze-link (context-only) resta invariato nel tuo file corrente
