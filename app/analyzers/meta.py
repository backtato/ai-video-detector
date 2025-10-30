# app/analyzers/meta.py
from __future__ import annotations
import subprocess, json
from typing import Dict, Any, Optional, List, Tuple

def _run(cmd: List[str]) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        return p.returncode, p.stdout or "", p.stderr or ""
    except Exception as e:
        return 1, "", str(e)

def _ffprobe_json(path: str) -> Dict[str, Any]:
    rc, out, err = _run(["ffprobe", "-hide_banner", "-v", "error",
                         "-print_format", "json", "-show_format", "-show_streams", path])
    if rc != 0:
        return {}
    try:
        return json.loads(out)
    except Exception:
        return {}

def detect_device(path: str) -> Dict[str, Any]:
    ffj = _ffprobe_json(path)
    vendor = None
    model  = None
    os_name = None

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

    md = _get("com.apple.quicktime.model", "model", "com.apple.quicktime.make")
    mk = _get("make", "com.apple.quicktime.make")
    sw = _get("software", "encoder", "com.apple.quicktime.software")
    an_mk = _get("com.android.manufacturer", "android.manufacturer", "manufacturer")
    an_md = _get("com.android.model", "android.model")

    vendor = mk or an_mk
    model  = md or an_md

    if sw and "iphone" in (sw.lower()):
        os_name = "iOS"
    elif vendor and any(x in vendor.lower() for x in ["samsung","xiaomi","huawei","google"]):
        os_name = "Android"

    return {"device": {"vendor": vendor, "model": model, "os": os_name}}
