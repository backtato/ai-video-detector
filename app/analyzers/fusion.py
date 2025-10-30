import os
from typing import Dict, Any, List, Tuple

REAL_TH = 0.35   # allineato alla UI conservativa
AI_TH   = 0.72
MIN_PEAK_SCORE = float(os.getenv("MIN_PEAK_SCORE", "0.12"))

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))

def _stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

def _local_peaks(xs: List[float], min_prom: float = None, min_dist: int = 2, top_k: int = 6) -> List[Tuple[int, float]]:
    # usa MIN_PEAK_SCORE come minimo, ma non scendere sotto 0.05
    if min_prom is None:
        min_prom = max(0.05, MIN_PEAK_SCORE)
    peaks = []
    n = len(xs)
    for i in range(1, n - 1):
        if xs[i] > xs[i - 1] and xs[i] > xs[i + 1]:
            left_min = min(xs[max(0, i - min_dist):i]) if i - min_dist >= 0 else xs[i - 1]
            right_min = min(xs[i + 1:min(n, i + 1 + min_dist)]) if i + 1 + min_dist <= n else xs[i + 1]
            prom = xs[i] - max(left_min, right_min)
            if prom >= min_prom:
                peaks.append((i, xs[i]))
    peaks.sort(key=lambda t: t[1], reverse=True)
    return peaks[:top_k]

def _reason_builder(hints: Dict[str, Any], video: Dict[str, Any], audio: Dict[str, Any],
                    fused: float, spread: float) -> str:
    parts = []
    comp = (hints or {}).get("compression")
    if not (video or {}).get("timeline"):
        parts.append("nessun frame video decodificato")

    dup_avg = (video.get("summary") or {}).get("dup_avg", 0.0) if video else 0.0
    flow = (hints or {}).get("flow_used", 0.0)
    motion = (hints or {}).get("motion_used", 0.0)

    if comp == "heavy":
        parts.append("compressione pesante")
    elif comp == "normal":
        parts.append("compressione normale")
    elif comp == "low":
        parts.append("compressione bassa")

    if dup_avg >= 0.65:
        if flow and flow > 1.0:
            parts.append("molti frame simili ma con micro-movimenti reali")
        else:
            parts.append("molti frame duplicati")

    flags = (audio or {}).get("flags_audio") or []
    if "likely_music" in flags:
        parts.append("audio con musica/ambiente")
    if "low_voice_presence" in flags:
        parts.append("voce scarsa/assente")

    v_tl = (video or {}).get("timeline_ai") or []
    a_tl = (audio or {}).get("timeline") or []
    if v_tl and a_tl:
        v_avg = _mean([x.get("ai_score", 0.5) for x in v_tl])
        a_avg = _mean([x.get("ai_score", 0.5) for x in a_tl])
        parts.append("segnali audio/video concordi" if abs(v_avg - a_avg) < 0.08 else "segnali misti audio/video")

    if (hints or {}).get("handheld_camera_likely", False):
        parts.append("handheld iPhone rilevato")

    if not parts:
        parts.append("segnali limitati")
    return " ; ".join(parts)

def _label_from_score(s: float) -> str:
    if s <= REAL_TH:
        return "real"
    if s >= AI_TH:
        return "ai"
    return "uncertain"

def fuse(meta: Dict[str, Any], hints: Dict[str, Any], video: Dict[str, Any], audio: Dict[str, Any]) -> Dict[str, Any]:
    v_tl = (video or {}).get("timeline_ai") or []
    a_tl = (audio or {}).get("timeline") or []

    T = max(len(v_tl), len(a_tl))
    if T == 0:
        T = max((video or {}).get("frames_sampled", 0), (audio or {}).get("frames_sampled", 0))
    if T == 0:
        fused = [0.5]
    else:
        def _get(lst: List[Dict[str, float]], i: int) -> float:
            if i < len(lst):
                return float(lst[i].get("ai_score", 0.5))
            return 0.5

        w_audio = 0.45
        w_video = 0.45
        boost = 0.10

        no_video = (not v_tl) or not (hints or {}).get("video_has_signal", False)
        if no_video:
            w_audio = 0.25
            w_video = 0.25
            fused = []
            for i in range(T):
                aa = _get(a_tl, i)
                fx = 0.5 + (aa - 0.5) * 0.30
                fused.append(_clamp(fx, 0.45, 0.55))
        else:
            fused = []
            for i in range(T):
                aa = _get(a_tl, i)
                va = _get(v_tl, i)
                fx = w_audio * aa + w_video * va + boost * ((va + aa) / 2 - 0.5)
                fused.append(_clamp(fx, 0.0, 1.0))

    fused_avg = _mean(fused)
    spread = _stdev(fused)

    bpp = (hints or {}).get("bpp") or 0.0
    comp = (hints or {}).get("compression")
    dup_avg = (video.get("summary") or {}).get("dup_avg", 0.0) if video else 0.0
    flow_avg = (hints or {}).get("flow_used", 0.0)
    motion_avg = (hints or {}).get("motion_used", 0.0)

    # Aggiustamenti conservativi
    adjust = 0.0
    if comp == "heavy" or bpp < 0.06:
        adjust += 0.06
    if dup_avg >= 0.65 and ((flow_avg and flow_avg > 1.0) or (motion_avg and motion_avg > 22.0)):
        adjust -= 0.04

    device = (meta or {}).get("device") or {}
    if (device.get("vendor") == "Apple" or device.get("os") == "iOS") and (hints or {}).get("video_has_signal", False):
        if (bpp >= 0.08) and (comp in (None, "normal", "low")) and ((flow_avg and flow_avg > 1.0) or (motion_avg and motion_avg >= 5.0)):
            adjust -= 0.05  # piÃ¹ favore al "real"

    fused_avg_adj = _clamp(fused_avg + adjust, 0.0, 1.0)

    base_conf = 0.20 + 2.5 * spread
    if comp == "heavy" or not (hints or {}).get("video_has_signal", True):
        base_conf *= 0.7
    conf = int(_clamp(base_conf, 0.10, 0.99) * 100)

    label = _label_from_score(fused_avg_adj)
    peaks = [{"t": i, "ai_score": v} for (i, v) in _local_peaks(fused, min_prom=None, min_dist=2, top_k=6)]
    reason = _reason_builder(hints, video or {}, audio or {}, fused_avg_adj, spread)

    return {
        "result": {
            "ai_score": round(fused_avg_adj, 6),
            "label": label,
            "confidence": conf,
            "reason": reason
        },
        "timeline_binned": [{"start": i, "end": i + 1, "ai_score": v} for i, v in enumerate(fused)],
        "peaks": peaks,
        "hints": hints or {}
    }