import os, tempfile
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import JSONResponse

from config import ENSEMBLE_WEIGHTS, CALIBRATION, MIN_FRAMES_FOR_CONFIDENCE, MIN_DURATION_SEC, THRESH_AI, THRESH_ORIGINAL
from calibration import calibrate, combine_scores
from detectors.metadata import ffprobe, score_metadata
from detectors.frame_artifacts import score_frame_artifacts
from detectors.audio import score_audio
from utils import extract_frames, video_duration_fps
from resolver import resolve_to_tempfile
from social_context import is_youtube_url, extract_youtube_id, youtube_context_score

app = FastAPI(title="AI Video Plausibility Detector (Resolver)")

def label_from(p: float) -> str:
    if p >= THRESH_AI: return "Likely AI"
    if p < THRESH_ORIGINAL: return "Likely Original"
    return "Inconclusive"

def run_pipeline(tmp_path: str):
    duration, fps, frame_count = video_duration_fps(tmp_path)
    info = ffprobe(tmp_path)
    md = score_metadata(info)
    frames = extract_frames(tmp_path, max_frames=64, stride= max(1, int(fps//6) if fps>0 else 4))
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
    confidence = 0.3 + 0.7 * min(1.0, (frame_count / 120.0))
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

@app.get("/")
def root():
    return {"status":"ok","service":"ai-video-detector"}

@app.get("/healthz")
def healthz():
    return {"ok": True}

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
        try: os.remove(tmp_path)
        except Exception: pass

@app.post("/analyze-url")
async def analyze_url(payload: dict = Body(...)):
    url = (payload.get("url") or "").strip()
    if not url:
        return JSONResponse({"error":"Missing url"}, status_code=400)
    try:
        tmp_path = resolve_to_tempfile(url)
    except Exception as e:
        return JSONResponse({"error": f"Resolver failed: {e}", "label":"Not assessable (content unavailable)"}, status_code=400)
    try:
        resp = run_pipeline(tmp_path)
        resp["notes"] = ["URL resolver used; only public/direct media resolved.", "MVP heuristics; replace with trained models."]
        resp["source_url"] = url
        return JSONResponse(resp)
    finally:
        try: os.remove(tmp_path)
        except Exception: pass

@app.post("/analyze-link")
async def analyze_link(payload: dict = Body(...)):
    """
    Context-only analysis for social links (YouTube supported in MVP).
    Returns a context score and label; NOT a pixel/content verdict.
    """
    url = (payload.get("url") or "").strip()
    if not url:
        return JSONResponse({"error":"Missing url"}, status_code=400)

    # YouTube
    if is_youtube_url(url):
        vid = extract_youtube_id(url)
        if not vid:
            return JSONResponse({"error":"Unable to extract YouTube videoId"}, status_code=400)
        try:
            ctx = youtube_context_score(vid)
        except Exception as e:
            return JSONResponse({"error": f"YouTube context failed: {e}"}, status_code=502)
        # map to response
        return JSONResponse({
            "context_only": True,
            "platform": ctx["platform"],
            "video_id": ctx["video_id"],
            "context_trust_score": ctx["score"],   # 0..1
            "context_label": ctx["context_label"], # "Context suggests authentic" / "Context suspicious" / "Context inconclusive"
            "context_signals": ctx["signals"],
            "label": "Not assessable (content unavailable)"
        })

    # Other platforms → unsupported in MVP
    return JSONResponse({"error":"Unsupported platform for context analysis (MVP supports YouTube)."}, status_code=400)
