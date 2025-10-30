# app/analyzers/meta.py
# Rilevamento "device" e presenza C2PA dai metadati. Zero dipendenze interne.

from __future__ import annotations
import subprocess
import json
from typing import Dict, Any, Optional, List, Tuple

def _run(cmd: List[str]) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        return p.returncode, p.stdout or "", p.stderr or ""
    except Exception as e:
        return 1, "", str(e)

def _ffprobe_json(path: str) -> Dict[str, Any]:
    rc, out, err = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    if rc != 0:
        return {}
    try:
        return json.loads(out) if out else {}
    except Exception:
        return {}

def _exiftool_json(path: str) -> Dict[str, Any]:
    rc, out, err = _run(["exiftool", "-json", "-struct", "-G1", path])
    if rc != 0:
        return {}
    try:
        data = json.loads(out or "[]")
        return data[0] if isinstance(data, list) and data else {}
    except Exception:
        return {}

def detect_device(path: str) -> Dict[str, Any]:
    """
    Ritorna: {"device": {"vendor": str|None, "model": str|None, "os": "iOS|Android|Unknown"}}
    Heuristics:
      - tag QuickTime: com.apple.quicktime.make/model => iOS
      - 'handler_name'/'encoder' contenenti iPhone/iPad/Apple => iOS
      - indizi Android (sm-, samsung, xiaomi, pixel, oneplus) => Android (debole)
    """
    ffj = _ffprobe_json(path)
    vendor = None
    model  = None
    os_name = "Unknown"

    fmt_tags = (ffj.get("format", {}) or {}).get("tags", {}) or {}
    streams  = ffj.get("streams", []) or []
    vtags = next((s.get("tags", {}) for s in streams if s.get("codec_type") == "video"), {})
    atags = next((s.get("tags", {}) for s in streams if s.get("codec_type") == "audio"), {})
    pool = [fmt_tags or {}, vtags or {}, atags or {}]

    def _get(*names: str) -> Optional[str]:
        for d in pool:
            for n in names:
                if n in d and d[n]:
                    return str(d[n])
        return None

    apple_make  = _get("com.apple.quicktime.make", "Make")
    apple_model = _get("com.apple.quicktime.model", "Model")
    handler     = _get("handler_name", "handler", "major_brand") or ""
    encoder     = _get("encoder", "com.apple.quicktime.software") or ""
    blob = (handler + " " + encoder).lower()

    if apple_make or apple_model:
        vendor = apple_make or "Apple"
        model  = apple_model
        os_name = "iOS"
    elif any(k in blob for k in ["iphone", "ipad", "apple"]):
        vendor = "Apple"
        model  = "iPhone/iPad?"
        os_name = "iOS"
    elif any(k in blob for k in ["android", "sm-", "samsung", "xiaomi", "oneplus", "redmi", "pixel"]):
        vendor = "Android?"
        model  = None
        os_name = "Android"

    return {"device": {"vendor": vendor, "model": model, "os": os_name}}

def detect_c2pa(path: str) -> Dict[str, Any]:
    """
    Ritorna: {"present": bool, "note": str}
    Heuristica veloce: cerca parole chiave C2PA/JUMBF/Content Credentials nell'XMP letto da exiftool.
    Se exiftool non ÃÂ¨ presente Ã¢ÂÂ present=False con nota.
    """
    probe = _exiftool_json(path)
    present = False
    note = ""

    if probe:
        try:
            text = json.dumps(probe, ensure_ascii=False).lower()
        except Exception:
            text = ""
        if any(k in text for k in ['"c2pa"', "content credentials", "jumbf", "manifest"]):
            present = True
            note = "Possibile C2PA/JUMBF nei metadati XMP"
    else:
        note = "ExifTool non disponibile o nessun XMP letto"

    return {"present": present, "note": note}