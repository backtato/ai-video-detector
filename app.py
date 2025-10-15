import os
import re
import tempfile
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import JSONResponse

from config import (
    ENSEMBLE_WEIGHTS,
    CALIBRATION,
    MIN_FRAMES_FOR_CONFIDENCE,
    MIN_DURATION_SEC,
    THRESH_AI,
    THRESH_ORIGINAL,
)
from calibration import calibrate, combine_scores
from detectors.metadata import ffprobe, score_metadata
from detectors.frame_artifacts import score_frame_artifacts
from detectors.audio import score_audio
from utils import extract_frames, video_duration_fps
from resolver import resolve_to_tempfile

# Context analyzer (YouTube)
from social_context import is_youtube_url, extract_youtube_id, youtube_context_score

app = FastAPI(title="AI Video Plausibility Detector (Resolver)")

# ------------------------- Verdict helpers -------------------------

def label_from(p: float) -> str:
    if p >= THRESH_AI:
        return "Likely AI"
    if p < THRESH_ORIGINAL:
        return "Likely Original"
    return "Inconclusive"

def run_pipeline(tmp_path: str):
    duration, fps, frame_count = video_duration_fps(tmp_path)
    info = ffprobe(tmp_path)

    md = score_metadata(info)
    frames = extract_frames(tmp_path, max_frames=64, stride=max(1, int(fps // 6) if fps else 4))
    fa = score_frame_artifacts(frames)
    au = score_audio(info)

    raw_weighted = {
        "metadata": (md["score"], ENSEMBLE_WEIGHTS["metadata"]),
        "frame_artifacts": (fa["score"], ENSEMBLE_WEIGHTS["frame_artifacts"]),
        "audio": (au["score"], ENSEMBLE_WEIGHTS["audio"]),
    }
    raw_score = combine_scores(raw_weighted)
    ai_plaus = calibrate(raw_score, CALIBRATION["a"], CALIBRATION["b"])

    enough_frames = frame_count >= MIN_FRAMES_FOR_CONFIDENCE and duration >= MIN_DURATION_SEC
    confidence = 0.3 + 0.7 * min(1.0, (frame_count / 120.0 if frame_count else 0.0))
    if not enough_frames:
        confidence *= 0.6

    return {
        "ai_plausibility": round(float(ai_plaus), 4),
        "confidence": round(float(confidence), 4),
        "label": label_from(float(ai_plaus)),
        "explanations": {
            "metadata": {"score": md["score"], "notes": md.get("notes", [])},
            "frame_artifacts": {"score": fa["score"], "notes": fa.get("notes", [])},
            "audio": {"score": au["score"], "notes": au.get("notes", [])},
        },
        "video_info": {"duration_sec": duration, "fps": fps, "frame_count": frame_count},
    }

# --------------------------- Health ---------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "ai-video-detector"}

@app.get("/healthz")
def healthz():
    return {"ok": True}

# ----------------------- Content analysis -----------------------

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[-1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        resp = run_pipeline(tmp_path)
        resp["notes"] = ["MVP heuristics; replace with trained models."]
        return JSONResponse(resp)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@app.post("/analyze-url")
async def analyze_url(payload: dict = Body(...)):
    url = (payload.get("url") or "").strip()
    if not url:
        return JSONResponse({"error": "Missing url"}, status_code=400)

    try:
        tmp_path = resolve_to_tempfile(url)
    except Exception as e:
        return JSONResponse(
            {"error": f"Resolver failed: {e}", "label": "Not assessable (content unavailable)"},
            status_code=400,
        )

    try:
        resp = run_pipeline(tmp_path)
        resp["notes"] = [
            "URL resolver used; only public/direct media resolved.",
            "MVP heuristics; replace with trained models.",
        ]
        resp["source_url"] = url
        return JSONResponse(resp)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# ----------------------- Context analysis (YouTube) -----------------------

YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "").strip()

@app.post("/analyze-link")
def analyze_link(payload: dict = Body(...)):
    """
    Context-only analysis (e.g., YouTube) without downloading content.
    Returns 'Not assessable (content unavailable)' for content verdict,
    plus a context_trust_score (0..1) and signals.
    """
    url = (payload.get("url") or "").strip()
    if not url:
        return JSONResponse({"error": "Missing url"}, status_code=400)

    if is_youtube_url(url):
        if not YOUTUBE_API_KEY:
            return JSONResponse(
                {"error": "Backend missing YOUTUBE_API_KEY env var",
                 "label": "Not assessable (content unavailable)"},
                status_code=500,
            )
        vid = extract_youtube_id(url)
        if not vid:
            return JSONResponse(
                {"error": "Cannot extract YouTube video id",
                 "label": "Not assessable (content unavailable)"},
                status_code=400,
            )
        # robust error handling: no 500s on YouTube failures
        try:
            ctx = youtube_context_score(vid, api_key=YOUTUBE_API_KEY)
        except ValueError as e:
            # Clean error from YouTube API (quota/restrictions/non-JSON)
            return JSONResponse(
                {"error": str(e), "label": "Not assessable (content unavailable)", "platform": "youtube"},
                status_code=502,  # external dependency error
            )
        except Exception as e:
            # Any unexpected error still must not bubble as 500
            return JSONResponse(
                {"error": f"Unexpected context error: {e}",
                 "label": "Not assessable (content unavailable)", "platform": "youtube"},
                status_code=502,
            )

        return JSONResponse({
            "label": "Not assessable (content unavailable)",
            "context_only": True,
            "platform": "youtube",
            "context_trust_score": round(ctx["score"] / 100.0, 4),
            "context_label": ctx["label"],
            "context_signals": ctx["signals"],
            "source_url": url,
        })

    # other platforms can be added (context-only) in future
    return JSONResponse(
        {"error": "Unsupported platform for analyze-link",
         "label": "Not assessable (content unavailable)"},
        status_code=400,
    )
