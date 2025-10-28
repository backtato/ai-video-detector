# app/analyzers/fusion.py
from typing import Dict, Any, List, Tuple
import math

# Soglie conservative allineate alla UI 1.1.3
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
    return cur if cur is not None else default

def _compress_index(meta: Dict[str, Any], video_stats: Dict[str, Any]) -> Tuple[float, Dict[str, str]]:
    """Ritorna (compression_index [0..1], notes). >0.6 = forte compressione."""
    notes = {}
    width  = _safe_get(meta, ["meta","width"],  _safe_get(meta, ["width"], 0))
    height = _safe_get(meta, ["meta","height"], _safe_get(meta, ["height"], 0))
    fps    = _safe_get(meta, ["meta","fps"],    _safe_get(meta, ["fps"], 0.0))
    br     = _safe_get(meta, ["meta","bit_rate"], _safe_get(meta, ["bit_rate"], 0))
    dup    = _safe_get(video_stats, ["summary","duplicate_ratio"], 0.0)

    # euristica bitrate per megapixel per frame
    if width and height and fps and br:
        mpf = (width*height)/1_000_000.0
        br_per_frame = br / max(fps,1.0)
        norm = br_per_frame / max(mpf, 0.1)  # kbit/s per MP per frame (scala grossolana)
        # soglie grezze per “molto compresso”
        if norm < 20000:   # molto basso
            c = 0.85
        elif norm < 40000: # basso
            c = 0.65
        elif norm < 80000: # medio
            c = 0.35
        else:
            c = 0.15
    else:
        c = 0.50  # ignoto → prudente

    # duplicate frames spingono compressione percepita
    dup = float(dup or 0.0)
    if dup >= 0.5:
        c = max(c, 0.8); notes["duplicate_frames"] = "Molti frame duplicati."
    elif dup >= 0.25:
        c = max(c, 0.6); notes["duplicate_frames"] = "Frame duplicati moderati."

    # fps molto basso
    if fps and fps < 18.0:
        c = max(c, 0.65); notes["low_fps"] = "Frame rate basso."

    # annotazioni sintetiche
    if c >= 0.75:
        notes["heavy_compression"] = "Compressione forte/WhatsApp-like."
    elif c >= 0.5:
        notes["moderate_compression"] = "Compressione moderata."
    return _clamp(c), notes

def _align_bins(n: int, vals: List[float]) -> List[float]:
    if n <= 0: return []
    if not vals: return [0.5]*n
    if len(vals) == n: return [float(x) for x in vals]
    if len(vals) > n:  return [float(x) for x in vals[:n]]
    last = float(vals[-1])
    return [float(x) for x in vals] + [last]*(n-len(vals))

def _label_from_score(s: float) -> str:
    if s <= THRESH_REAL: return "real"
    if s >= THRESH_AI:   return "ai"
    return "uncertain"

def fuse(video_stats: Dict[str, Any], audio_stats: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusion conservativa:
      - pesi: audio 0.65, video 0.25, micro-boost 0.10 (direzionale all'audio)
      - attenuazione compressione (pull verso 0.5) 0.05..0.10
      - confidenza da spread temporale + qualità + audio
    """
    v = video_stats or {}
    a = audio_stats or {}
    m = meta or {}

    # Timeline lunghezza (per-secondo)
    duration = _safe_get(m, ["meta","duration"], _safe_get(m, ["duration"], 0.0)) or 0.0
    n_bins   = max(int(duration)+1, 1)

    video_tl = _align_bins(n_bins, _safe_get(v, ["timeline","ai_score"], []))
    audio_tl = _align_bins(n_bins, _safe_get(a, ["timeline","ai_score"], []))

    # score medi
    v_mean = float(_safe_get(v, ["summary","ai_score_mean"], 0.5))
    a_mean = float(_safe_get(a, ["scores","tts_like"], 0.5))  # audio “verso AI”: più alto = più AI-like

    # compressione
    comp_idx, comp_notes = _compress_index(m, v)

    # fusione frame-wise
    fused = []
    for i in range(n_bins):
        base = 0.65*audio_tl[i] + 0.25*video_tl[i] + 0.10*(audio_tl[i]-0.5) + 0.50  # re-centering
        base = _clamp(base, 0.0, 1.0)
        # attenuazione compressione → pull verso 0.5
        pull = 0.05 + 0.05 * (1.0 if comp_idx >= 0.75 else (0.5 if comp_idx >= 0.5 else 0.0))
        base = 0.5 + (base - 0.5) * (1.0 - pull)
        fused.append(_clamp(base))

    # score globale = media robusta
    ai_score = float(sum(fused)/len(fused)) if fused else 0.5
    label = _label_from_score(ai_score)

    # confidenza: spread + contributo audio + qualità
    # spread = deviazione dalla neutralità temporale
    dev = sum(abs(x-0.5) for x in fused)/max(len(fused),1)
    conf = 0.10 + 0.70*min(dev*2.0, 1.0) + 0.10*abs(a_mean-0.5)*2.0 + 0.10*(1.0 - comp_idx)
    confidence = int(round(_clamp(conf, 0.10, 0.99)*100))

    # reason
    reasons = []
    if comp_idx >= 0.75:
        reasons.append("Compressione pesante (WhatsApp/ri-upload)")
    elif comp_idx >= 0.5:
        reasons.append("Compressione moderata")
    dup = _safe_get(v, ["summary","duplicate_ratio"], 0.0) or 0.0
    if dup >= 0.5:
        reasons.append("Molti frame duplicati")
    flow = _safe_get(v, ["summary","flow_mean"], 0.0) or 0.0
    if flow <= 0.05:
        reasons.append("Movimento scarso/innaturale")
    tts_like = float(a_mean)
    hnr_proxy = float(_safe_get(a, ["scores","hnr_proxy"], 0.6))
    if tts_like >= 0.35:
        reasons.append("Profilo audio omogeneo/robotico (tts-like)")
    if hnr_proxy <= 0.55:
        reasons.append("Basso rapporto armonico-rumore")
    if not reasons:
        reasons.append("Segnali misti o neutri")

    timeline_binned = [{"start": float(i), "end": float(i+1), "ai_score": float(fused[i])} for i in range(n_bins)]
    # picchi “informativi”: scarta valori ~0.5
    peaks = [{"t": i, "ai_score": float(s)} for i,s in enumerate(fused) if s <= 0.35 or s >= 0.72]

    result = {
        "label": label,
        "ai_score": float(round(ai_score, 4)),
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
