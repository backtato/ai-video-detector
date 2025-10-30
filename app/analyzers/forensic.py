# app/analyzers/forensic.py
# Estrazione forense leggera via ExifTool + euristica presenza C2PA/JUMBF.
# Non dipende da librerie Python extra (usa subprocess).

import json
import subprocess

def _run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)

def exiftool_json(path: str):
    p = _run(["exiftool", "-json", "-struct", "-G1", path])
    try:
        data = json.loads(p.stdout or "[]")
        return data[0] if isinstance(data, list) and data else {}
    except Exception:
        return {}

def c2pa_present_from_exif(exif: dict) -> bool:
    # Heuristica: cerca indicatori C2PA/JUMBF/manifest claim nell'output ExifTool
    try:
        text = json.dumps(exif).lower()
    except Exception:
        return False
    return ("c2pa" in text) or ("jumbf" in text) or ("manifest" in text and "claim" in text)

def analyze(path: str) -> dict:
    ex = exiftool_json(path)
    return {
        "exif": {"has_data": bool(ex), "subset": {k: ex.get(k) for k in list(ex.keys())[:30]}},
        "c2pa": {"present": c2pa_present_from_exif(ex)}
    }