# app/fusion.py
from typing import Dict, Any, List
import numpy as np

def _safe(v, d=0.5):
    try:
        if v is None: return d
        return float(v)
    except Exception:
        return d

def _bin_timeline(timeline: List[dict], duration: float, bin_sec: float = 1.0, mode: str = "max") -> List[dict]:
    """
    Binning temporale semplice di una timeline [{start,end,ai_score}].
    """
    if not timeline:
        if duration and duration > 0:
            return [{"start": 0.0, "end": min(duration, 1.0), "ai_score": 0.5}]
        return [{"start": 0.0, "end": 1.0, "ai_score": 0.5}]

    if not duration or duration <= 0:
        duration = max([seg.get("end", 0.0) for seg in timeline] + [1.0])

    bins = []
    t = 0.0
    while t < duration:
        te = min(t + bin_sec, duration)
        # prendere max (o media) dei segmenti che intersecano [t, te]
        vals = []
        for seg in timeline:
            s0, s1 = seg.get("start", 0.0), seg.get("end", 0.0)
            if s1 <= t or s0 >= te:
                continue
            vals.append(float(seg.get("ai_score", 0.5)))
        if not vals:
            vals = [0.5]
        bins.append({"start": float(t), "end": float(te), "ai_score": float(max(vals) if mode=="max" else np.mean(vals))})
        t = te
    return bins

def _dynamic_bin(bins: List[dict]) -> List[dict]:
    """
    Densifica i bin se la varianza locale è alta (>0.05).
    Semplice: spezza in due i bin con (ai_score dev) > soglia.
    """
    if not bins:
        return bins
    out: List[dict] = []
    for b in bins:
        score = _safe(b.get("ai_score"), 0.5)
        if abs(score - 0.5) > 0.05 and (b["end"] - b["start"]) > 1.0:
            mid = (b["start"] + b["end"]) / 2.0
            out.append({"start": b["start"], "end": mid, "ai_score": score})
            out.append({"start": mid, "end": b["end"], "ai_score": score})
        else:
            out.append(b)
    return out

def _peaks(bins: List[dict], min_dev: float = 0.15, min_coverage: float = 0.10) -> List[dict]:
    """
    Identifica segmenti con |score-0.5|>min_dev per >=10% di durata totale (regola semplificata).
    """
    if not bins: return []
    total = sum(b["end"] - b["start"] for b in bins)
    if total <= 0: return []
    # Merge molto semplice di bin “alti”
    high = [b for b in bins if abs(_safe(b["ai_score"],0.5)-0.5) > min_dev]
    if not high: return []
    # Accorpa contigui
    peaks: List[dict] = []
    cur = None
    for b in high:
        if cur is None:
            cur = dict(b)
        else:
            if abs(b["start"] - cur["end"]) < 1e-6:
                cur["end"] = b["end"]
                cur["ai_score"] = max(cur["ai_score"], b["ai_score"])
            else:
                peaks.append(cur); cur = dict(b)
    if cur: peaks.append(cur)
    # Filtra per copertura
    peaks = [p for p in peaks if (p["end"] - p["start"]) / total >= min_coverage]
    return peaks

def fuse_scores(frame: float, audio: float, hints: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusione pesata + etichetta + confidenza e reason codes.
    - ai_score = 0.6*frame + 0.3*audio + 0.1*hints_score
    - label: <0.35 reale, 0.35–0.65 incerto, >0.65 AI
    """
    f = _safe(frame, 0.5)
    a = _safe(audio, 0.5)

    # Hints score semplice
    hs = 0.5
    reasons: List[str] = []
    if hints.get("no_c2pa", False):
        reasons.append("no_c2pa")
        hs += 0.05
    if hints.get("apple_quicktime_tags", False):
        reasons.append("iphone_quicktime_tags")
        hs -= 0.02
    if hints.get("editor_like", False):
        reasons.append("editing_software_tags")
        hs += 0.05
    if hints.get("low_motion", False):
        reasons.append("low_motion")
        hs += 0.03
    hs = float(np.clip(hs, 0.0, 1.0))

    ai_score = 0.6*f + 0.3*a + 0.1*hs
    ai_score = float(np.clip(ai_score, 0.0, 1.0))

    if ai_score < 0.35:
        label = "real"
    elif ai_score > 0.65:
        label = "ai"
    else:
        label = "uncertain"

    return {
        "ai_score": ai_score,
        "label": label,
        "reasons": reasons
    }

def finalize_with_timeline(fusion_core: Dict[str, Any], bins: List[dict]) -> Dict[str, Any]:
    """
    Aggiunge confidenza e calcola peaks dalla timeline binned.
    """
    if not bins:
        conf = 0.5
        peaks = []
    else:
        vals = [float(b["ai_score"]) for b in bins]
        stdev = float(np.std(vals)) if vals else 0.0
        conf = float(np.clip(1.0 - stdev, 0.0, 1.0))
        peaks = _peaks(bins, min_dev=0.15, min_coverage=0.10)

    out = dict(fusion_core)
    out["confidence"] = conf
    return out, peaks

def build_hints(meta: Dict[str, Any], video_flags: List[str], c2pa_present: bool, dev_fingerprint: Dict[str,Any]) -> Dict[str, Any]:
    """
    Converte meta/flags in un dizionario di hint booleani.
    """
    hints = {
        "no_c2pa": not bool(c2pa_present),
        "apple_quicktime_tags": bool(dev_fingerprint.get("apple_quicktime_tags")),
        "editor_like": bool(dev_fingerprint.get("editor_like")),
        "low_motion": "low_motion" in (video_flags or []),
    }
    return hints
