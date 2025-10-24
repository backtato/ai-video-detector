from typing import List, Dict, Any
import numpy as np

def _safe_avg(xs: List[float]) -> float:
    if not xs: return 0.5
    return float(np.clip(np.mean(xs), 0.0, 1.0))

def fuse_and_label(meta: Dict[str,Any],
                   forensic: Dict[str,Any],
                   v_scores: List[float], v_timeline: List[dict],
                   a_scores: List[float], a_timeline: List[dict]) -> Dict[str,Any]:
    w_video, w_audio = 0.6, 0.4
    if a_timeline and len(a_timeline)==1 and a_timeline[0]["ai_score"]==0.5 and len(a_scores)==1:
        w_video, w_audio = 0.8, 0.2

    v_mean = _safe_avg(v_scores)
    a_mean = _safe_avg(a_scores)
    ai_score = float(np.clip(w_video*v_mean + w_audio*a_mean, 0.0, 1.0))

    timeline = []
    timeline.extend(v_timeline)
    timeline.extend(a_timeline)

    if ai_score < 0.35: label = "Con alta probabilità è REALE"; conf = 0.7
    elif ai_score > 0.65: label = "Con alta probabilità è AI"; conf = 0.7
    else: label = "Esito incerto / Parziale"; conf = 0.6

    return {
        "ok": True,
        "meta": {
            "width": meta.get("width"),
            "height": meta.get("height"),
            "fps": meta.get("fps"),
            "duration": meta.get("duration"),
            "bit_rate": meta.get("bit_rate"),
            "vcodec": meta.get("vcodec"),
            "acodec": meta.get("acodec"),
            "format_name": meta.get("format_name"),
        },
        "forensic": forensic,
        "scores": {
            "frame": v_mean,
            "audio": a_mean
        },
        "fusion": {
            "ai_score": ai_score,
            "label": label,
            "confidence": conf
        },
        "timeline": sorted(timeline, key=lambda s: (s["start"], s["end"]))
    }