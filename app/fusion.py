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
    if not timeline:
        if duration and duration>0:
            return [{"start":0.0,"end":min(duration,1.0),"ai_score":0.5}]
        return [{"start":0.0,"end":1.0,"ai_score":0.5}]
    if not duration or duration<=0:
        duration = max([seg.get("end",0.0) for seg in timeline] + [1.0])

    out = []
    t = 0.0
    while t < duration:
        te = min(t+bin_sec, duration)
        vals = []
        for seg in timeline:
            s0, s1 = seg.get("start",0.0), seg.get("end",0.0)
            if s1 <= t or s0 >= te: continue
            vals.append(float(seg.get("ai_score",0.5)))
        if not vals: vals=[0.5]
        out.append({"start":float(t), "end":float(te), "ai_score": float(max(vals) if mode=="max" else np.mean(vals))})
        t = te
    return out

def _dynamic_bin(bins: List[dict]) -> List[dict]:
    if not bins: return bins
    out = []
    for b in bins:
        s = _safe(b.get("ai_score"), 0.5)
        if abs(s-0.5)>0.05 and (b["end"]-b["start"])>1.0:
            mid = (b["start"]+b["end"])/2.0
            out.append({"start": b["start"], "end": mid, "ai_score": s})
            out.append({"start": mid, "end": b["end"], "ai_score": s})
        else:
            out.append(b)
    return out

def _peaks(bins: List[dict], min_dev: float=0.15, min_coverage: float=0.10) -> List[dict]:
    if not bins: return []
    total = sum(b["end"]-b["start"] for b in bins)
    if total<=0: return []
    high = [b for b in bins if abs(_safe(b.get("ai_score"),0.5)-0.5)>min_dev]
    if not high: return []
    peaks = []
    cur = None
    for b in high:
        if cur is None: cur = dict(b)
        else:
            if abs(b["start"]-cur["end"])<1e-6:
                cur["end"] = b["end"]
                cur["ai_score"] = max(cur["ai_score"], b["ai_score"])
            else:
                peaks.append(cur); cur = dict(b)
    if cur: peaks.append(cur)
    peaks = [p for p in peaks if (p["end"]-p["start"])/total >= min_coverage]
    return peaks

def build_hints(meta: Dict[str,Any], video_flags: List[str], c2pa_present: bool, dev_fp: Dict[str,Any]) -> Dict[str,Any]:
    return {
        "no_c2pa": not bool(c2pa_present),
        "apple_quicktime_tags": bool(dev_fp.get("apple_quicktime_tags")),
        "editor_like": bool(dev_fp.get("editor_like")),
        "low_motion": "low_motion" in (video_flags or []),
    }

def fuse_scores(frame: float, audio: float, hints: Dict[str,Any]) -> Dict[str,Any]:
    f, a = _safe(frame,0.5), _safe(audio,0.5)
    hs = 0.5
    reasons = []
    if hints.get("no_c2pa"): reasons.append("no_c2pa"); hs += 0.05
    if hints.get("apple_quicktime_tags"): reasons.append("iphone_quicktime_tags"); hs -= 0.02
    if hints.get("editor_like"): reasons.append("editing_software_tags"); hs += 0.05
    if hints.get("low_motion"): reasons.append("low_motion"); hs += 0.03
    hs = float(np.clip(hs, 0.0, 1.0))

    ai_score = float(np.clip(0.6*f + 0.3*a + 0.1*hs, 0.0, 1.0))
    if ai_score < 0.35: label = "real"
    elif ai_score > 0.65: label = "ai"
    else: label = "uncertain"
    return {"ai_score": ai_score, "label": label, "reasons": reasons}

def finalize_with_timeline(fusion_core: Dict[str,Any], bins: List[dict]) -> Dict[str,Any]:
    if not bins:
        conf, peaks = 0.5, []
    else:
        vals = [float(b["ai_score"]) for b in bins]
        stdev = float(np.std(vals)) if vals else 0.0
        conf = float(np.clip(1.0 - stdev, 0.0, 1.0))
        peaks = _peaks(bins, min_dev=0.15, min_coverage=0.10)
    out = dict(fusion_core)
    out["confidence"] = conf
    return out, peaks
