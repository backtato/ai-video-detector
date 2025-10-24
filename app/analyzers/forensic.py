from typing import Dict, Any

def forensic_checks(ffprobe_json: Dict[str,Any]) -> Dict[str,Any]:
    fmt = (ffprobe_json or {}).get("format", {}) or {}
    streams = (ffprobe_json or {}).get("streams", []) or []
    v = next((s for s in streams if s.get("codec_type")=="video"), {})
    tags = {}
    tags.update(fmt.get("tags") or {})
    tags.update(v.get("tags") or {})

    c2pa_present = any(k.lower().startswith("c2pa") or "adobe" in k.lower() for k in tags.keys()) \
                   or any("c2pa" in str(val).lower() for val in tags.values())

    quicktime_apple = any(
        k.lower().startswith("com.apple.quicktime") or
        (isinstance(val,str) and "apple" in val.lower())
        for k,val in tags.items()
    )

    return {
        "c2pa": {"present": bool(c2pa_present)},
        "apple_quicktime_tags": bool(quicktime_apple),
        "flags": []
    }