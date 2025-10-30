import json, subprocess

def exiftool_json(path: str):
    try:
        out = subprocess.check_output(["exiftool","-json","-struct","-G1",path], text=True, stderr=subprocess.DEVNULL, timeout=20)
        data = json.loads(out or "[]")
        return data[0] if isinstance(data, list) and data else {}
    except Exception:
        return {}

def c2pa_present(exif: dict) -> bool:
    try:
        t = json.dumps(exif).lower()
    except Exception:
        return False
    return ("c2pa" in t) or ("jumbf" in t) or ("manifest" in t and "claim" in t)

def detect_device(exif: dict) -> str | None:
    for k in ("QuickTime:Make","QuickTime:Model","EXIF:Make","EXIF:Model"):
        v = exif.get(k)
        if v: return str(v)
    return None

def forensic_summary(path: str):
    ex = exiftool_json(path)
    return {
        "c2pa": {"present": c2pa_present(ex)},
        "exif_quick": {k: ex.get(k) for k in ("QuickTime:Make","QuickTime:Model","EXIF:Make","EXIF:Model") if k in ex}
    }