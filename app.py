import os, tempfile, json
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from config import ENSEMBLE_WEIGHTS, CALIBRATION, MIN_FRAMES_FOR_CONFIDENCE, MIN_DURATION_SEC
from calibration import calibrate, combine_scores
from detectors.metadata import ffprobe, score_metadata
from detectors.frame_artifacts import score_frame_artifacts
from detectors.audio import score_audio
from utils import extract_frames, video_duration_fps

app = FastAPI(title="AI Video Plausibility Detector (MVP)")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Save to temp file
    suffix = os.path.splitext(file.filename)[-1] if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        duration, fps, frame_count = video_duration_fps(tmp_path)
        info = ffprobe(tmp_path)

        # Detectors
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

        # confidence: scale by duration and frame count
        enough_frames = frame_count >= MIN_FRAMES_FOR_CONFIDENCE and duration >= MIN_DURATION_SEC
        confidence = 0.3 + 0.7 * min(1.0, (frame_count / 120.0))  # cap at ~120 frames
        if not enough_frames:
            confidence *= 0.6

        resp = {
            "ai_plausibility": round(float(ai_plaus), 4),
            "confidence": round(float(confidence), 4),
            "explanations": {
                "metadata": {"score": md["score"], "notes": md.get("notes", [])},
                "frame_artifacts": {"score": fa["score"], "notes": fa.get("notes", [])},
                "audio": {"score": au["score"], "notes": au.get("notes", [])},
            },
            "video_info": {"duration_sec": duration, "fps": fps, "frame_count": frame_count},
            "notes": [
                "MVP heuristics only—replace with trained models before production.",
                "Probability is calibrated via a simple logistic function; recalibrate on your dataset."
            ]
        }

        return JSONResponse(resp)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
