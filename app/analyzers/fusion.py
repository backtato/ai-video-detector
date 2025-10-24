from typing import List, Dict, Any
import numpy as np
import math

def _safe_avg(xs: List[float]) -> float:
    if not xs: return 0.5
    return float(np.clip(np.mean(xs), 0.0, 1.0))

def _bin_timeline(timeline: List[dict], duration: float, bin_sec: float = 1.0, mode: str = "max") -> List[dict]:
    if duration <= 0: duration = max([seg["end"] for seg in timeline] + [0.0])
    bins = []
    nbins = max(int(math.ceil(duration / bin_sec)), 1)
    for i in range(nbins):
        start = i * bin_sec
        end = min((i + 1) * bin_sec, duration)
        vals = []
        for seg in timeline:
            # overlap?
            if seg["end"] > start and seg["start"] < end:
                vals.append(seg["ai_score"])
        if not vals:
            score = 0.5
        else:
            score = max(vals) if mode == "max" else float(np.mean(vals))
        bins.append({"start": float(start), "end": float(end), "ai_score": float(np.clip(score, 0.0, 1.0))})
    return bins

def _top_peaks(bins: List[dict], k: int = 3) -> List[dict]:
    # restituisce i k bin con ai_score più alto
    order = sorted(bins, key=lambda b: b["ai_score"], reverse=True)
    return order[:k]

def fuse_and_label(meta: Dict[str,Any],
                   forensic: Dict[str,Any],
                   v_scores: List[float], v_timeline: List[dict],
                   a_scores: List[float], a_timeline: List[dict]) -> Dict[str,Any]:
    # pesi base
    w_video, w_audio = 0.6, 0.4
    if a_timeline and len(a_timeline)==1 and a_timeline[0]["ai_score"]==0.5 and len(a_scores)==1:
        w_video, w_audio = 0.8, 0.2

    v_mean = _safe_avg(v_scores)
    a_mean = _safe_avg(a_scores)
    ai_score = float(np.clip(w_video*v_mean + w_audio*a_mean, 0.0, 1.0))

    # timeline grezza (video + audio)
    timeline = []
    timeline.extend(v_timeline)
    timeline.extend(a_timeline)
    timeline = sorted(timeline, key=lambda s: (s["start"], s["end"]))

    # timeline binned a 1s per la UI
    duration = float(meta.get("duration") or 0.0)
    timeline_binned = _bin_timeline(timeline, duration, bin_sec=1.0, mode="max")
    peaks = _top_peaks(timeline_binned, k=3)

    # soglie conservative
    if ai_score < 0.35: label = "Con alta probabilità è REALE"; conf = 0.7
    elif ai_score > 0.65: label = "Con alta probabilità è AI"; conf = 0.7
    else: label = "Esito incerto / Parziale"; conf = 0.6

    return {
        "ok": True,
        "meta": {
            "width": meta.get("width"),
            "height": meta.get("height"),
            "fps": meta.get("fps"),
            "duration": duration,
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
        # tieni anche la timeline dettagliata per debug
        "timeline": timeline,
        # aggiunte UI-friendly
        "timeline_binned": timeline_binned,
        "peaks": peaks
    }