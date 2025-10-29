# app/analyzers/fusion.py
from typing import Dict, Any, List, Tuple

THRESH_REAL = 0.35
THRESH_AI   = 0.72

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))

def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def _score_from_tracks(v: Dict[str, Any], a: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    # pesi conservativi: audio 0.65, video 0.25, piccolo boost direzionale 0.10
    # clamp verso 0.5 se qualità bassa (“compression pull”)
    va = float(v.get("summary", {}).get("ai_score_avg", 0.0) or 0.0)  # se disponibile
    # fallback: media timeline_ai
    if va == 0.0:
        tl = v.get("timeline_ai") or []
        if tl:
            va = sum(float(x.get("ai_score", 0.0) or 0.0) for x in tl) / len(tl)
        else:
            va = 0.5

    aa = 0.5
    at = a.get("timeline") or []
    if at:
        aa = sum(float(x.get("ai_score", 0.5) or 0.5) for x in at) / len(at)
    else:
        aa = float(a.get("scores", {}).get("audio_mean", 0.5) or 0.5)

    w_audio = 0.65
    w_video = 0.25
    sd = (aa - 0.5) * 0.10  # small directional
    s = _clamp(w_audio * aa + w_video * va + sd, 0.0, 1.0)

    hints = {}
    # qualità/compressione dai hints video
    vdup  = float(v.get("summary", {}).get("dup_avg", 0.0) or 0.0)
    vbloc = float(v.get("summary", {}).get("blockiness_avg", 0.0) or 0.0)
    vband = float(v.get("summary", {}).get("banding_avg", 0.0) or 0.0)
    flow  = float(v.get("summary", {}).get("optflow_mag_avg", 0.0) or 0.0)
    motion= float(v.get("summary", {}).get("motion_avg", 0.0) or 0.0)
    bpp   = float(_safe_get(a, ["scores","bpp"], 0.0) or 0.0)

    heavy_comp = (vbloc > 0.35 or vband > 0.35)
    if heavy_comp:
        s = 0.5 + (s - 0.5) * 0.95  # leggera trazione verso neutro
        hints["heavy_compression"] = True

    # confidenza: base dallo “spread” più bonus audio − penalità compressione
    spread = abs(aa - va)
    conf = 0.50 + spread * 0.35
    if heavy_comp:
        conf -= 0.05
    conf = int(_clamp(conf, 0.10, 0.99) * 100)

    # reason sintetico
    reason = []
    if heavy_comp:
        reason.append("compressione moderata/pesante")
    if flow >= 1.5 and motion < 0.2:
        reason.append("flow alto con motion basso")
    if not reason:
        reason.append("segnali misti/neutri")

    return s, {"confidence": conf, "reason": "; ".join(reason)}

def _label(s: float) -> str:
    if s <= THRESH_REAL:
        return "real"
    if s >= THRESH_AI:
        return "ai"
    return "uncertain"

def fuse(video_stats: Dict[str, Any], audio_stats: Dict[str, Any], meta: Dict[str, Any], c2pa: Dict[str, Any]) -> Dict[str, Any]:
    s, info = _score_from_tracks(video_stats or {}, audio_stats or {})

    # timeline binned: se disponibile timeline video al secondo, altrimenti piatta
    duration = int(float(meta.get("duration", 0.0) or 0.0) + 0.5)
    if duration <= 0:
        duration = int(float(video_stats.get("duration", 0.0) or 0.0) + 0.5)

    timeline_binned: List[Dict[str, Any]] = []
    if duration > 0:
        for t in range(duration):
            timeline_binned.append({"start": t, "end": t+1, "ai_score": float(s)})

    peaks = [{"t": t, "ai_score": float(s)} for t in range(min(duration, 16))]  # limite UI

    v = video_stats or {}
    m = meta or {}
    hints = {
        "bpp": float(_safe_get(m, ["bit_rate"], 0) or 0) / max((int(_safe_get(m, ["width"], 0) or 0) * int(_safe_get(m, ["height"], 0) or 0) * float(_safe_get(m, ["fps"], 0.0) or 1.0)), 1.0),
        "compression": "low" if not info.get("reason","").startswith("compressione") else "moderate",
        "video_has_signal": True if v else False,
        "flow_used": float(_safe_get(v, ["summary","optflow_mag_avg"], 0.0) or 0.0),
        "motion_used": float(_safe_get(v, ["summary","motion_avg"], 0.0) or 0.0),
        "w": int(_safe_get(v, ["width"], 0) or _safe_get(m, ["width"], 0) or 0),
        "h": int(_safe_get(v, ["height"], 0) or _safe_get(m, ["height"], 0) or 0),
        "fps": float(_safe_get(m, ["fps"], 0.0) or 0.0),
        "br": int(_safe_get(m, ["bit_rate"], 0) or 0),
    }
    if c2pa and c2pa.get("present"):
        hints["c2pa_present"] = "Manifest/Content Credentials rilevati"

    result = {
        "label": _label(s),
        "ai_score": round(float(s), 6),
        "confidence": info.get("confidence", 70),
        "reason": info.get("reason", "segnali misti/neutri")
    }

    return {
        "result": result,
        "timeline_binned": timeline_binned,
        "peaks": peaks,
        "hints": hints,
    }
