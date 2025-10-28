# app/analyzers/fusion.py
from typing import Dict, Any, List, Tuple
import math

# Soglie conservative allineate alla UI 1.1.3
THRESH_REAL = 0.35
THRESH_AI   = 0.75  # ↑ più conservativa

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))

def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur if cur is not None else default

def _compress_index(meta: Dict[str, Any], video_stats: Dict[str, Any]) -> Tuple[float, Dict[str, str]]:
    """Ritorna (compression_index [0..1], notes). >0.6 = forte compressione."""
    notes = {}
    width  = _safe_get(meta, ["meta","width"],  _safe_get(meta, ["width"], 0))
    height = _safe_get(meta, ["meta","height"], _safe_get(meta, ["height"], 0))
    br     = _safe_get(meta, ["meta","bit_rate"], _safe_get(meta, ["bit_rate"], 0))
    vcodec = str(_safe_get(meta, ["meta","vcodec"], "")).lower()
    fmt    = str(_safe_get(meta, ["meta","format_name"], "")).lower()

    px = float(width) * float(height)
    if px <= 0.0 or br is None:
        return 0.5, notes

    # bitrate per pixel al secondo (grezzo)
    # soglie grezze: < 0.04 = forte compressione; 0.04–0.12 = media; >0.12 = bassa
    # (tarate su h264 1080p/4k consumer)
    bpp = float(br) / max(1.0, px * max(1.0, float(_safe_get(meta, ["meta","fps"], 30.0))))
    if bpp < 0.04:
        idx = 0.85; notes["compression"] = "heavy"
    elif bpp < 0.12:
        idx = 0.55; notes["compression"] = "medium"
    else:
        idx = 0.15; notes["compression"] = "light"

    if "whatsapp" in fmt or "3gp" in fmt:  # hint molto grezzo
        idx = max(idx, 0.75)
        notes["messaging_like"] = "possible"

    return float(_clamp(idx, 0.0, 1.0)), notes

def _align_bins(n: int, tl: List[Dict[str, float]]) -> List[float]:
    arr = [0.5] * n
    if not tl:
        return arr
    for i in range(min(n, len(tl))):
        v = tl[i].get("ai_score", 0.5)
        arr[i] = _clamp(v)
    return arr

def _label_from_score(s: float) -> str:
    if s <= THRESH_REAL:
        return "real"
    if s >= THRESH_AI:
        return "ai"
    return "uncertain"

def fuse(m: Dict[str, Any], v: Dict[str, Any], a: Dict[str, Any]) -> Dict[str, Any]:
    """Fusione conservativa audio+video con pull da compressione e gating audio."""
    # Numero di bin per-secondo: prova a ricavarlo dalla timeline audio, altrimenti video
    n_bins = 0
    if isinstance(_safe_get(a, ["timeline"], None), list):
        n_bins = max(n_bins, len(a["timeline"]))
    if isinstance(_safe_get(v, ["timeline"], None), list):
        n_bins = max(n_bins, len(v["timeline"]))
    if n_bins <= 0:
        n_bins = int(math.ceil(float(_safe_get(m, ["meta","duration"], 0.0)) or 1.0))
        n_bins = max(1, min(n_bins, 180))

    # allinea timeline
    video_tl = _align_bins(n_bins, _safe_get(v, ["timeline"], []))
    audio_tl = _align_bins(n_bins, _safe_get(a, ["timeline"], []))

    comp_idx, comp_notes = _compress_index(m, v)

    fused = []

    # ---- Gating/pesi dinamici audio ----
    w_audio_base, w_video = 0.55, 0.40
    # mean audio features
    a_mean      = float(_safe_get(a, ["scores","tts_like"], 0.5))
    hnr_proxy   = float(_safe_get(a, ["scores","hnr_proxy"], 0.6))
    if hnr_proxy >= 0.62 and a_mean <= 0.45:
        w_audio = 0.40
        w_video = 0.50
    else:
        w_audio = w_audio_base
    w_boost = 0.10
    total_w = w_audio + w_video + w_boost
    w_audio /= total_w; w_video /= total_w; w_boost /= total_w

    for i in range(n_bins):
        a_i = float(audio_tl[i])
        v_i = float(video_tl[i]) if video_tl else 0.5
        base = (w_audio * a_i) + (w_video * v_i) + (w_boost * (a_i - 0.5) + 0.5)
        base = _clamp(base, 0.0, 1.0)

        # pull da compressione
        pull = 0.05 + 0.05 * (1.0 if comp_idx >= 0.75 else (0.5 if comp_idx >= 0.5 else 0.0))
        base = 0.5 + (base - 0.5) * (1.0 - pull)

        fused.append(_clamp(base))

    ai_score = float(sum(fused)/len(fused)) if fused else 0.5

    # Bias soft pro-reale per device mobile + alta qualità
    device_txt = str(_safe_get(m, ["meta","format_name"], "")).lower()
    dev_obj = _safe_get(m, ["device"], {})
    if isinstance(dev_obj, dict):
        device_txt += " " + (str(dev_obj.get("vendor","")) + " " + str(dev_obj.get("model",""))).lower()

    is_apple = ("apple" in device_txt) or ("quicktime" in device_txt) or ("mov" in device_txt)
    if is_apple and comp_idx <= 0.25 and ai_score >= 0.55:
        ai_score = max(0.0, ai_score - 0.07)

    ai_score = _clamp(ai_score)
    label    = _label_from_score(ai_score)

    # confidenza: dispersione + punteggi audio “forti” + bassa compressione
    dev  = sum(abs(x-0.5) for x in fused)/max(len(fused),1)
    conf = 0.10 + 0.70*min(dev*2.0, 1.0) + 0.10*max(0.0, (abs(a_mean-0.5)*2.0 - 0.2)) + 0.10*(1.0 - comp_idx)
    confidence = int(round(_clamp(conf, 0.10, 0.99)*100))

    # timeline_binned + peaks (come prima)
    timeline_binned = [{"start": i, "end": i+1, "ai_score": float(fused[i])} for i in range(len(fused))]
    # picchi veri (ignora quasi-neutri)
    peaks = [{"t": i, "ai_score": float(fused[i])} for i in range(len(fused)) if fused[i] <= 0.30 or fused[i] >= 0.70]

    # ragioni
    reasons: List[str] = []
    if comp_idx >= 0.75:
        reasons.append("Compressione pesante (WhatsApp/ri-upload)")

    dup = float(_safe_get(v, ["summary","dup_avg"], 0.0) or 0.0)
    if dup >= 0.85:
        reasons.append("Molti frame molto simili (scena statica)")

    flow = float(_safe_get(v, ["summary","optflow_mag_avg"], 0.0) or 0.0)  # FIX chiave
    motion_avg = float(_safe_get(v, ["summary","motion_avg"], 0.0) or 0.0)
    if flow <= 0.05 and motion_avg <= 0.5:
        reasons.append("Movimento scarso/innaturale")

    tts_like = float(a_mean)
    if tts_like >= 0.50:  # ↑ più severa
        reasons.append("Profilo audio omogeneo/robotico (tts-like)")

    if hnr_proxy <= 0.55:
        reasons.append("Basso rapporto armonico-rumore")

    if not reasons:
        reasons.append("Segnali misti o neutri")

    result = {
        "label": label,
        "ai_score": float(ai_score),
        "confidence": int(confidence),
        "reason": "; ".join(reasons),
    }

    hints = {}
    hints.update(comp_notes)
    if _safe_get(m, ["forensic","c2pa","present"], False):
        hints["c2pa_present"] = "Manifest/Content Credentials rilevati"

    return {
        "result": result,
        "timeline_binned": timeline_binned,
        "peaks": peaks,
        "hints": hints,
    }