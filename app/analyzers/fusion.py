from typing import Dict, Any, List, Tuple
import math

REAL_TH = 0.35
AI_TH   = 0.72

def _clamp(v, lo, hi): 
    return max(lo, min(hi, v))

def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))

def _stdev(xs: List[float]) -> float:
    if len(xs) < 2: 
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)

def _smooth(xs: List[float], w: int = 3) -> List[float]:
    if w <= 1 or len(xs) <= 2:
        return xs[:]
    half = w // 2
    out = []
    for i in range(len(xs)):
        a = max(0, i - half)
        b = min(len(xs), i + half + 1)
        out.append(_mean(xs[a:b]))
    return out

def _local_peaks(xs: List[float], min_prom: float = 0.04, min_dist: int = 2, top_k: int = 6) -> List[Tuple[int, float]]:
    n = len(xs)
    peaks = []
    for i in range(1, n - 1):
        if xs[i] > xs[i - 1] and xs[i] >= xs[i + 1]:
            left_min = min(xs[max(0, i - 10):i]) if i > 0 else xs[i]
            right_min = min(xs[i + 1:min(n, i + 11)]) if i + 1 < n else xs[i]
            prom = xs[i] - max(left_min, right_min)
            if prom >= min_prom:
                peaks.append((i, xs[i], prom))
    peaks.sort(key=lambda t: t[1], reverse=True)
    filtered = []
    for p in peaks:
        if all(abs(p[0] - q[0]) >= min_dist for q in filtered):
            filtered.append(p)
        if len(filtered) >= top_k:
            break
    return [(i, v) for (i, v, _) in filtered]

def _mean_from_series(series: List[Dict[str, Any]], key="ai_score") -> float:
    if not series:
        return 0.5
    return _mean([float(b.get(key, 0.5)) for b in series])

def _reason_builder(hints: Dict[str, Any], video: Dict[str, Any], audio: Dict[str, Any], fused_avg: float, spread: float) -> str:
    parts = []
    comp = hints.get("compression")
    if comp == "heavy":
        parts.append("compressione pesante")
    elif comp == "moderate":
        parts.append("compressione moderata")

    dup = (video.get("summary") or {}).get("dup_avg")
    if isinstance(dup, (int, float)) and dup >= 0.6:
        parts.append("molti frame duplicati")

    v_avg = _mean_from_series(video.get("timeline_ai", []))
    a_avg = _mean_from_series(audio.get("timeline", []))
    dv = abs(v_avg - a_avg)
    if dv < 0.07:
        parts.append("segnali audio/video concordi")
    else:
        parts.append("segnali misti audio/video")

    if spread < 0.04:
        parts.append("variazioni minime nel tempo")
    elif spread > 0.12:
        parts.append("variazioni marcate nel tempo")

    return " ; ".join(parts) if parts else "segnali neutri"

def fuse(video: Dict[str, Any], audio: Dict[str, Any], hints: Dict[str, Any]) -> Dict[str, Any]:
    # Serie per-secondo
    v_series = [b.get("ai_score", 0.5) for b in video.get("timeline_ai", [])]
    a_series = [b.get("ai_score", 0.5) for b in audio.get("timeline", [])]

    L = min(len(v_series), len(a_series)) if v_series and a_series else max(len(v_series), len(a_series))
    if L == 0:
        return {
            "result": {"ai_score": 0.5, "label": "uncertain", "confidence": 35, "reason": "dati insufficienti"},
            "timeline_binned": [],
            "peaks": [],
            "hints": hints or {}
        }
    v_series = (v_series[:L] if v_series else [0.5] * L)
    a_series = (a_series[:L] if a_series else [0.5] * L)

    v_s = _smooth(v_series, 3)
    a_s = _smooth(a_series, 3)

    # pesi conservativi
    w_audio = 0.65
    w_video = 0.25

    fused = []
    for va, aa in zip(v_s, a_s):
        boost = 0.10 if ((va - 0.5) * (aa - 0.5) > 0) else 0.0
        fx = w_audio * aa + w_video * va + boost * ((va + aa) / 2 - 0.5)
        fused.append(_clamp(fx, 0.0, 1.0))

    fused_avg = _mean(fused)
    spread = _stdev(fused)

    # penalità da hints
    bpp = hints.get("bpp") or 0.0
    comp = hints.get("compression")
    dup_avg = (video.get("summary") or {}).get("dup_avg", 0.0)

    penalty = 0.0
    if comp == "heavy" or bpp < 0.06:
        penalty += 0.06
    elif comp == "moderate" or bpp < 0.10:
        penalty += 0.03
    if isinstance(dup_avg, (int, float)) and dup_avg >= 0.6:
        penalty += 0.03

    fused_avg_pen = _clamp(fused_avg - 0.5 * penalty, 0.0, 1.0)

    conf_base = _clamp(0.15 + 2.3 * spread, 0.10, 0.99)
    conf = conf_base - penalty * 0.4
    conf = int(round(_clamp(conf, 0.10, 0.99) * 100))

    if fused_avg_pen <= REAL_TH:
        label = "real"
    elif fused_avg_pen >= AI_TH:
        label = "ai"
    else:
        label = "uncertain"

    peaks = [{"t": i, "ai_score": v} for (i, v) in _local_peaks(fused, min_prom=0.05, min_dist=2, top_k=6)]

    reason = _reason_builder(hints, video, audio, fused_avg_pen, spread)

    return {
        "result": {
            "ai_score": round(fused_avg_pen, 6),
            "label": label,
            "confidence": conf,
            "reason": reason
        },
        "timeline_binned": [{"start": i, "end": i + 1, "ai_score": v} for i, v in enumerate(fused)],
        "peaks": peaks,
        "hints": hints or {}
    }
