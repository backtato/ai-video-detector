# app/analyzers/meta.py
import os
import json
import subprocess
from typing import Dict, Any, Optional

def _run(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)

def _ffprobe_json(path: str) -> Dict[str, Any]:
    """
    Esegue ffprobe e ritorna il JSON completo dei metadati.
    """
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
    """
    Esegue exiftool e ritorna il primo oggetto JSON (se presente).
    """
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
    Estrae metadati “ricchi” da ffprobe + exiftool (fallback se disponibile).
    Restituisce un dizionario con campi utili alla UI e alle euristiche.
    """
    info = _ffprobe_json(path)
    fmt = info.get("format", {}) or {}
    streams = info.get("streams", []) or []
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    a = next((s for s in streams if s.get("codec_type") == "audio"), {})

    # Base meta
    def _as_int(val) -> Optional[int]:
        try:
            return int(val)
        except Exception:
            return None

    def _as_float(val) -> Optional[float]:
        try:
            return float(val)
        except Exception:
            return None

    width = _as_int(v.get("width"))
    height = _as_int(v.get("height"))
    duration = _as_float(fmt.get("duration"))
    bit_rate = _as_int(fmt.get("bit_rate"))

    # FPS robusto
    fps = None
    for key in ("avg_frame_rate", "r_frame_rate"):
        r = v.get(key)
        if r and isinstance(r, str) and "/" in r and r != "0/0":
            try:
                n, d = r.split("/")
                n, d = float(n), float(d)
                if d != 0:
                    fps = n / d
                    if fps > 0:
                        break
            except Exception:
                pass
    if not fps:
        fps_num = v.get("nb_frames")
        if fps_num and duration:
            try:
                fps = float(fps_num) / float(duration)
            except Exception:
                fps = None

    # Tag
    tags = {}
    tags.update(fmt.get("tags") or {})
    tags.update(v.get("tags") or {})
    tags.update(a.get("tags") or {})

    # EXIFTOOL (opzionale)
    exif = _exiftool_json(path)
    if exif:
        # preferisci exiftool per questi campi se presenti
        for k in ["Make", "Model", "Software", "CreateDate", "ModifyDate", "GPSLatitude", "GPSLongitude", "Duration"]:
            if k in exif and exif[k] is not None:
                tags.setdefault(k, exif[k])

    # Normalizzazione output
    meta = {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "bit_rate": bit_rate,
        "vcodec": v.get("codec_name") or None,
        "acodec": a.get("codec_name") or None,
        "format_name": fmt.get("format_name") or None,
        # Copriamo casi utili per UI / reason
        "make": tags.get("Make") or tags.get("com.apple.quicktime.make") or tags.get("com.android.manufacturer"),
        "model": tags.get("Model") or tags.get("com.apple.quicktime.model") or tags.get("com.android.model"),
        "software": tags.get("Software") or tags.get("encoder") or tags.get("com.apple.quicktime.software") or v.get("codec_tag_string"),
        "creation_time": tags.get("creation_time") or tags.get("CreateDate") or tags.get("com.apple.quicktime.creationdate"),
        "orientation": tags.get("Orientation") or tags.get("rotate") or v.get("tags", {}).get("rotate"),
        "gps": {
            "lat": tags.get("GPSLatitude"),
            "lon": tags.get("GPSLongitude"),
        },
        "raw_tags": tags,
    }
    return meta

def detect_device_fingerprint(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deduce informazioni indicative sul device/software in base ai tag.
    Non pretende accuratezza assoluta—serve come “hint”.
    """
    tags = (meta or {}).get("raw_tags") or {}

    # Apple/QuickTime hints
    apple_qt = any(
        str(k).lower().startswith("com.apple.quicktime") or ("apple" in str(v).lower())
        for k, v in tags.items()
    )
    iphone_like = False
    model = (meta.get("model") or "") or ""
    if isinstance(model, str) and model.lower().startswith(("iphone", "ipad", "ipod")):
        iphone_like = True

    android_like = False
    make = (meta.get("make") or "") or ""
    if isinstance(make, str) and any(x in make.lower() for x in ["samsung", "xiaomi", "google", "oneplus", "huawei", "oppo"]):
        android_like = True

    editor_like = False
    software = (meta.get("software") or "") or ""
    if isinstance(software, str) and any(x in software.lower() for x in ["premiere", "after effects", "davinci", "capcut", "resolve", "ffmpeg"]):
        editor_like = True

    return {
        "apple_quicktime_tags": bool(apple_qt),
        "iphone_like": bool(iphone_like),
        "android_like": bool(android_like),
        "editor_like": bool(editor_like),
    }

def detect_c2pa(path: str) -> Dict[str, Any]:
    """
    Controllo basilare C2PA: cerca tag noti via exiftool e nel JSON di ffprobe.
    """
    present = False

    # 1) exiftool: cerca riferimenti a C2PA / Adobe Content Credentials
    p = _run(["exiftool", "-j", path])
    txt = (p.stdout or "") + (p.stderr or "")
    if any(s in txt.lower() for s in ["c2pa", "content credentials", "adobe signature"]):
        present = True

    # 2) ffprobe tags alla ricerca di c2pa
    info = _ffprobe_json(path)
    fmt = info.get("format", {}) or {}
    streams = info.get("streams", []) or []
    tags = {}
    tags.update(fmt.get("tags") or {})
    for s in streams:
        tags.update(s.get("tags") or {})
    if any("c2pa" in str(k).lower() or "c2pa" in str(v).lower() for k, v in tags.items()):
        present = True

    return {"present": bool(present)}