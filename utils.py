import json
import subprocess
from typing import Tuple

def _run(cmd: list[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err

def ffprobe_json(path: str) -> dict:
    code, out, err = _run([
        "ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path
    ])
    if code != 0:
        raise RuntimeError(f"ffprobe failed: {err.strip()}")
    return json.loads(out)

def parse_fps(streams: list[dict]) -> float:
    # trova il primo stream video
    for s in streams:
        if s.get("codec_type") == "video":
            r = s.get("r_frame_rate") or s.get("avg_frame_rate") or "0/1"
            try:
                n, d = r.split("/")
                n, d = float(n), float(d) if float(d) != 0 else 1.0
                return max(1e-6, n / d)
            except Exception:
                return 0.0
    return 0.0

def video_duration_fps(meta: dict) -> tuple[float, float, int]:
    fmt = meta.get("format", {})
    dur = float(fmt.get("duration", 0.0))
    fps = parse_fps(meta.get("streams", []))
    nb = 0
    for s in meta.get("streams", []):
        if s.get("codec_type") == "video":
            nb = int(s.get("nb_frames", 0) or 0)
            break
    return dur, fps, nb
