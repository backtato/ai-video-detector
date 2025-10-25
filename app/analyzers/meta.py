# app/analyzers/meta.py
import json
import subprocess
from typing import Dict, Any, Optional

def _run(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)

def _ffprobe_json(path: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams", "-show_chapters",
        path
    ]
    p = _run(cmd)
    try:
        return json.loads(p.stdout or "{}")
    except Exception:
        return {}

def _exiftool_json(path: str) -> Dict[str, Any]:
    p = _run(["exiftool", "-j", "-n", path])
    try:
        data = json.loads(p.stdout or "[]")
        if isinstance(data, list) and data:
            return data[0]
    except Exception:
        pass
    return {}

def extract_metadata(path: str) -> Dict[str, Any]:
    """
    Estrae metadati 'ricchi' (ffprobe + exiftool se presente).
    """
    info = _ffprobe_json(path)
    fmt = info.get("format", {}) or {}
    streams = info.get("streams", []) or []
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    a = next((s for s in streams if s.get("codec_type") == "audio"), {})

    def _as_int(x): 
        try: return int(x)
        except: return None

    def _as_float(x):
        try: return float(x)
        except: return None

    width  = _as_int(v.get("width"))
    height = _as_int(v.get("height"))
    duration = _as_float(fmt.get("duration"))
    bit_rate = _as_int(fmt.get("bit_rate"))

    # FPS robusto
    fps = None
    for k in ("avg_frame_rate","r_frame_rate"):
        r = v.get(k)
        if r and isinstance(r,str) and "/" in r and r != "0/0":
            try:
                n,d = r.split("/")
                n,d = float(n), float(d)
                if d!=0: fps = n/d; break
            except: pass
    if not fps:
        nb = v.get("nb_frames")
        if nb and duration:
            try: fps = float(nb)/float(duration)
            except: fps = None

    tags = {}
    tags.update(fmt.get("tags") or {})
    tags.update(v.get("tags") or {})
    tags.update(a.get("tags") or {})

    exif = _exiftool_json(path)
    if exif:
        for k in ["Make","Model","Software","CreateDate","ModifyDate","GPSLatitude","GPSLongitude","Duration"]:
            if k in exif and exif[k] is not None:
                tags.setdefault(k, exif[k])

    meta = {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "bit_rate": bit_rate,
        "vcodec": v.get("codec_name") or None,
        "acodec": a.get("codec_name") or None,
        "format_name": fmt.get("format_name") or None,
        "make": tags.get("Make") or tags.get("com.apple.quicktime.make") or tags.get("com.android.manufacturer"),
        "model": tags.get("Model") or tags.get("com.apple.quicktime.model") or tags.get("com.android.model"),
        "software": tags.get("Software") or tags.get("encoder") or tags.get("com.apple.quicktime.software"),
        "creation_time": tags.get("creation_time") or tags.get("CreateDate") or tags.get("com.apple.quicktime.creationdate"),
        "orientation": tags.get("Orientation") or tags.get("rotate") or v.get("tags", {}).get("rotate"),
        "gps": {"lat": tags.get("GPSLatitude"), "lon": tags.get("GPSLongitude")},
        "raw_tags": tags,
        "ffprobe_raw": info,  # utile per debug forense
    }
    return meta

def detect_device_fingerprint(meta: Dict[str, Any]) -> Dict[str, Any]:
    tags = (meta or {}).get("raw_tags") or {}
    apple_qt = any(
        str(k).lower().startswith("com.apple.quicktime") or ("apple" in str(v).lower())
        for k,v in tags.items()
    )
    model = str(meta.get("model") or "")
    make  = str(meta.get("make") or "")
    sw    = str(meta.get("software") or "")

    return {
        "apple_quicktime_tags": bool(apple_qt),
        "iphone_like": model.lower().startswith(("iphone","ipad","ipod")),
        "android_like": any(b in make.lower() for b in ["samsung","xiaomi","google","oneplus","huawei","oppo"]),
        "editor_like": any(x in sw.lower() for x in ["premiere","after effects","capcut","resolve","davinci","ffmpeg"]),
    }

def detect_c2pa(path: str) -> Dict[str, Any]:
    present = False
    # exiftool scan
    p = _run(["exiftool","-j", path])
    raw = (p.stdout or "") + (p.stderr or "")
    if any(s in raw.lower() for s in ["c2pa","content credentials","adobe signature"]):
        present = True
    return {"present": bool(present)}
