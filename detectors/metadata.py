import json
import subprocess
from typing import Dict, Any

def ffprobe(path: str) -> Dict[str, Any]:
    """Run ffprobe and return parsed JSON metadata."""
    cmd = [
        "ffprobe", "-v", "error", "-hide_banner",
        "-print_format", "json",
        "-show_format", "-show_streams",
        path
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffprobe failed")
    return json.loads(proc.stdout or "{}")

def score_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Heuristics:
    - Missing encoder / creation_time increases AI suspicion slightly.
    - Non-physical FPS or very constant GOP length may increase suspicion.
    Returns:
      dict(score=float[0..1], details=dict(...))
    """
    details = {}
    streams = meta.get("streams", [])
    format_tags = meta.get("format", {}).get("tags", {}) or {}

    has_encoder = bool(format_tags.get("encoder") or format_tags.get("major_brand"))
    has_creation = bool(format_tags.get("creation_time") or format_tags.get("date"))
    fps_weird = False
    gop_constant = False

    for st in streams:
        if st.get("codec_type") == "video":
            r = st.get("r_frame_rate") or st.get("avg_frame_rate") or "0/1"
            try:
                num, den = r.split("/")
                fps = (float(num) / float(den)) if float(den) != 0 else 0.0
            except Exception:
                fps = 0.0
            # Unusually high FPS suggests re-encoding from generator / screen capture
            if fps >= 120.0 or fps <= 5.0:
                fps_weird = True

            # GOP constancy proxy: if both nb_frames and duration_ts present and ratio is near-integer
            nb_frames = st.get("nb_frames")
            time_base = st.get("time_base")
            if nb_frames and time_base:
                # If time_base exists, not a strong signal; placeholder:
                pass

    # Simple scoring: start neutral 0.5, subtract for real-world signals, add for oddities
    score = 0.5
    if not has_encoder:
        score += 0.1
    if not has_creation:
        score += 0.05
    if fps_weird:
        score += 0.1
    if gop_constant:
        score += 0.05

    score = max(0.0, min(1.0, score))
    details.update({
        "has_encoder": has_encoder,
        "has_creation_time": has_creation,
        "fps_weird": fps_weird,
        "gop_constant": gop_constant,
    })
    return {"score": score, "details": details}
