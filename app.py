import os
import re
import tempfile
import requests
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
        resp["notes"] = [
            "MVP heuristics; replace with trained models.",
        ]
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
        # Non scaricabile legalmente o non è un media pubblico
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

def extract_youtube_id(url: str):
    # ?v=ID
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    # youtu.be/ID
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    # youtube.com/shorts/ID
    m = re.search(r"youtube\.com/shorts/([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    return None

def youtube_context_score(video_id: str) -> dict:
    # 1) Video details
    vi = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={"part": "snippet,contentDetails,statistics,status", "id": video_id, "key": YOUTUBE_API_KEY},
        timeout=10,
    ).json()
    items = vi.get("items", [])
    if not items:
        return {"score": 50, "signals": ["Video not found or private"], "platform": "youtube"}

    v = items[0]
    snip = v.get("snippet", {})
    ch_id = snip.get("channelId")
    title = (snip.get("title") or "").lower()
    description = (snip.get("description") or "").lower()

    # 2) Channel stats (heuristics)
    subs = 0
    ch_age_years = 0.0
    if ch_id:
        ch = requests.get(
            "https://www.googleapis.com/youtube/v3/channels",
            params={"part": "snippet,statistics,status", "id": ch_id, "key": YOUTUBE_API_KEY},
            timeout=10,
        ).json()
        chi = (ch.get("items") or [])
        if chi:
            c = chi[0]
            subs = int(c.get("statistics", {}).get("subscriberCount", 0) or 0)
            published = c.get("snippet", {}).get("publishedAt", "")
            try:
                from datetime import datetime, timezone
                if published:
                    dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                    ch_age_years = max(0.0, (datetime.now(timezone.utc) - dt).days / 365.25)
            except Exception:
                pass

    # 3) Heuristic context score 0..100
    score = 50
    signals = []
    # audience signals
    if subs > 1_000_000:
        score += 10; signals.append("Large audience channel")
    elif subs > 100_000:
        score += 6; signals.append("Significant audience channel")
    elif subs < 1_000:
        score -= 5; signals.append("Very small audience")
    # age signals
    if ch_age_years >= 5:
        score += 6; signals.append(f"Old channel (~{ch_age_years:.1f}y)")
    elif ch_age_years < 0.5:
        score -= 6; signals.append("Very new channel")
    # content hints
    clickbait_terms = ["shocking", "you won't believe", "gone wrong", "impossible", "ai generated"]
    if any(t in title for t in clickbait_terms):
        score -= 5; signals.append("Clickbait-like title")
    if " ai " in f" {title} " or " ai " in f" {description} ":
        signals.append("Mentions AI in title/description")

    score = max(0, min(100, score))
    return {"score": score, "signals": signals, "platform": "youtube"}

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

    if ("youtube.com" in url) or ("youtu.be" in url):
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
        ctx = youtube_context_score(vid)
        return JSONResponse({
            "label": "Not assessable (content unavailable)",
            "context_trust_score": round(ctx["score"] / 100.0, 4),
            "context_signals": ctx["signals"],
            "platform": ctx["platform"],
            "source_url": url,
        })

    # Other platforms can be added here later (Instagram/TikTok APIs) – context-only.
    return JSONResponse(
        {"error": "Unsupported platform for analyze-link",
         "label": "Not assessable (content unavailable)"},
        status_code=400,
    )
