# detectors/metadata.py
import json
import subprocess
from typing import Dict, Any

def ffprobe(path: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=codec_type,codec_name,avg_frame_rate",
        "-of", "json",
        path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        return {"format": {}, "streams": []}
    try:
        data = json.loads(out.stdout or "{}")
    except Exception:
        data = {"format": {}, "streams": []}
    return data

def score_metadata(meta: Dict[str, Any]) -> float:
    """
    Heuristica MVP:
    - frame rate estremi o inconsistenti alzano il sospetto
    - assenza traccia audio riduce "naturalezza"
    """
    streams = meta.get("streams", [])
    fmt = meta.get("format", {})
    dur = float(fmt.get("duration", 0) or 0)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    frs = []
    for s in streams:
        if s.get("codec_type") == "video":
            afr = s.get("avg_frame_rate") or "0/1"
            try:
                num, den = afr.split("/")
                frs.append(float(num) / float(den) if float(den) != 0 else 0.0)
            except Exception:
                pass
    fr = (sum(frs) / len(frs)) if frs else 0.0

    score = 0.5
    if not has_audio:
        score += 0.1
    if fr < 12 or fr > 90:
        score += 0.1
    if dur < 2:
        score += 0.05
    return max(0.0, min(1.0, score))
