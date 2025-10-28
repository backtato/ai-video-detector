# app/analyzers/fusion.py
from typing import Dict, Any, List, Tuple
import math

# Soglie UI-friendly (pro-real)
THRESH_REAL = 0.35
THRESH_AI   = 0.75

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))

def _get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur if cur is not None else default

def _label(s: float) -> str:
    if s <= THRESH_REAL: return "real"
    if s >= THRESH_AI:   return "ai"
    return "uncertain"

def _compression_from_meta(meta: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    bpp = bit_rate / (w*h*fps)
    heavy  < 0.04
    medium 0.04–0.12
    light  > 0.12
    """
    notes: Dict[str, Any] = {}
    w   = float(_get(meta, ["meta","width"],  _get(meta, ["width"], 0)) or 0)
    h   = float(_get(meta, ["meta","height"], _get(meta, ["height"], 0)) or 0)
    fps = float(_get(meta, ["meta","fps"],    _get(meta, ["fps"], 0)) or 0)
    br  = float(_get(meta, ["meta","bit_rate"], _get(meta, ["bit_rate"], 0)) or 0)
    px  = w*h
    if px <= 0 or fps <= 0 or br <= 0:
        notes.update({"compression":"unknown","bpp":None,"w":int(w),"h":int(h),"fps":fps,"br":int(br)})
        return 0.50, notes
    bpp = br / (px*fps)
    notes.update({"bpp": round(bpp,5), "w": int(w), "h": int(h), "fps": fps, "br": int(br)})
    if bpp < 0.04:
        notes["compression"] = "heavy";  idx = 0.85
    elif bpp < 0.12:
        notes["compression"] = "medium"; idx = 0.55
    else:
        notes["compression"] = "light";  idx = 0.15
    return float(idx), notes

def _align_bins(n: int, arr: List[float]) -> List[float]:
    if n <= 0: return []
    if not arr: return [0.5]*n
    if len(arr) == n: return [float(x) for x in arr]
    if len(arr) > n:  return [float(x) for x in arr[:n]]
    out = [0.5]*n
    out[:len(arr)] = [float(x) for x in arr]
    return out

def _video_score_from_metrics(v: Dict[str, Any], n_bins: int) -> List[float]:
    """
    'video ai_score' sintetico da dup/blockiness/banding/motion
    (placeholder ragionevole finché non c'è un modello video ML)
    """
    tl = _get(v, ["timeline"], []) or []
    if not tl or not isinstance(tl, list):
        return [0.5]*n_bins
    scores = []
    for i in range(min(n_bins, len(tl))):
        sec = tl[i] or {}
        dup  = float(sec.get("dup", 0.0))
        blk  = float(sec.get("blockiness", 0.0))
        band = float(sec.get("banding", 0.0))   # ~0.45 baseline
        mot  = float(sec.get("motion", 0.0))
        blk_n  = max(0.0, min(1.0, blk*8.0))            # 0..~0.2 ⇒ 0..1
        band_n = max(0.0, min(1.0, (band-0.45)*10.0))   # baseline 0.45
        mot_n  = max(0.0, min(1.0, mot/40.0))           # 40 = “molto movimento”
        v_ai = 0.25*dup + 0.25*blk_n + 0.20*band_n + 0.30*(1.0 - mot_n)
        scores.append(_clamp(v_ai))
    return _align_bins(n_bins, scores)

def fuse(video_stats: Dict[str, Any], audio_stats: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    v = video_stats or {}
    a = audio_stats or {}
    m = meta or {}

    duration = float(_get(m, ["meta","duration"], _get(v, ["duration"], 0.0)) or 0.0)
    n_bins   = max(int(math.ceil(duration)), 1)

    # timeline audio
    a_tl = _get(a, ["timeline"], []) or []
    a_bins = _align_bins(n_bins, [float(x.get("ai_score", 0.5)) for x in a_tl] if a_tl else [])

    # timeline video (preferisci timeline_ai se già calcolata)
    v_ai_tl = _get(v, ["timeline_ai"], [])
    if v_ai_tl and isinstance(v_ai_tl, list) and all(isinstance(x, dict) and "ai_score" in x for x in v_ai_tl):
        v_bins = _align_bins(n_bins, [float(x["ai_score"]) for x in v_ai_tl])
    else:
        v_bins = _video_score_from_metrics(v, n_bins)

    video_has_signal = any(abs(x-0.5) > 0.03 for x in v_bins)

    # indici globali
    comp_idx, comp_notes = _compression_from_meta(m)
    flow        = float(_get(v, ["summary","optflow_mag_avg"], 0.0) or 0.0)
    motion_avg  = float(_get(v, ["summary","motion_avg"],      0.0) or 0.0)
    dup_avg     = float(_get(v, ["summary","dup_avg"],         0.0) or 0.0)

    # audio “naturale”?
    tts_like  = float(_get(a, ["scores","tts_like"],  0.5))
    hnr_proxy = float(_get(a, ["scores","hnr_proxy"], 0.6))
    audio_looks_natural = (hnr_proxy >= 0.60 and tts_like <= 0.45)

    # pesi base (riduciamo il boost audio)
    w_audio_base, w_video = 0.50, 0.45
    if audio_looks_natural:
        w_audio = 0.38; w_video = 0.54   # se audio “umano”, diamo più fiducia al video
    else:
        w_audio = w_audio_base
    w_boost = 0.07  # prima 0.10: abbassato
    total_w = w_audio + w_video + w_boost
    w_audio /= total_w; w_video /= total_w; w_boost /= total_w

    natural_motion = (flow >= 0.30) or (motion_avg >= 5.0)
    not_heavy      = comp_idx <= 0.60

    fused: List[float] = []
    conflict_bins = 0

    for i in range(n_bins):
        a_i = a_bins[i] if i < len(a_bins) else 0.5
        v_i = v_bins[i] if i < len(v_bins) else 0.5

        # attenuazione “per-bin” del segnale audio quando il video contraddice (<0.55)
        # e la scena è naturale e non heavy: l’audio non deve sovrastare il video.
        if natural_motion and not_heavy and v_i < 0.55 and a_i > 0.65:
            conflict_bins += 1
            # attenuazione proporzionale a quanto il video è basso (max -40%)
            atten = min(0.40, (0.55 - v_i) * 0.8)  # v_i=0.35 -> atten≈0.16
            a_i = 0.5 + (a_i - 0.5) * (1.0 - atten)

        base = (w_audio * a_i) + (w_video * v_i) + (w_boost * (a_i - 0.5) + 0.5)
        base = _clamp(base)

        # pull da compressione
        pull = 0.05 + 0.05 * (1.0 if comp_idx >= 0.75 else (0.5 if comp_idx >= 0.5 else 0.0))
        base = 0.5 + (base - 0.5) * (1.0 - pull)

        # CAP quando il video è neutro (nessun segnale) ma moto naturale:
        if natural_motion and not_heavy and (not video_has_signal) and a_i >= 0.65:
            base = min(base, 0.80)

        fused.append(_clamp(base))

    # Gating “globale” se molti bin sono in conflitto (audio alto vs video basso)
    conflict_ratio = conflict_bins / float(max(1, n_bins))
    if natural_motion and not_heavy and conflict_ratio >= 0.25:
        # abbassa la media (pro-real) e limita i picchi oltre 0.85
        fused = [min(x, 0.85) for x in fused]
        # lieve spinta additiva verso 0.5
        fused = [0.5 + (x-0.5)*0.92 for x in fused]  # 8% in meno di deviazione

    ai_score = float(sum(fused)/len(fused)) if fused else 0.5
    label    = _label(ai_score)

    # confidenza: dipende dalla “deviazione” + qualità audio + minor compressione
    dev  = sum(abs(x-0.5) for x in fused)/max(len(fused),1)
    conf = 0.10 + 0.70*min(dev*2.0, 1.0) + 0.10*max(0.0, (abs(tts_like-0.5)*2.0 - 0.2)) + 0.10*(1.0 - comp_idx)
    confidence = int(round(_clamp(conf, 0.10, 0.99)*100))

    # timeline & peaks
    timeline_binned = [{"start": i, "end": i+1, "ai_score": float(fused[i])} for i in range(len(fused))]
    peaks = [{"t": i, "ai_score": float(fused[i])} for i in range(len(fused)) if fused[i] <= 0.30 or fused[i] >= 0.70]

    # reason string
    reasons: List[str] = []
    if comp_idx >= 0.75:
        reasons.append("Compressione pesante (WhatsApp/ri-upload)")
    elif comp_idx >= 0.50:
        reasons.append("Compressione moderata")
    if dup_avg >= 0.90:
        reasons.append("Molti frame quasi identici (scena statica)")
    if flow <= 0.10 and motion_avg <= 5.0:
        reasons.append("Movimento scarso/innaturale")

    # Reason per conflitto audio-vs-video
    # (si attiva quando abbiamo effettivamente attenuato o fatto gating)
    if natural_motion and not_heavy and (conflict_ratio >= 0.20):
        reasons.append("Audio dominante senza supporto del video")

    # C2PA (path corretto)
    c2pa_present = bool(_get(m, ["forensic","c2pa","present"], False))
    hints = dict(comp_notes)
    if c2pa_present:
        hints["c2pa_present"] = True

    hints["flow_used"] = round(flow,4)
    hints["motion_used"] = round(motion_avg,4)
    hints["video_has_signal"] = bool(video_has_signal)
    hints["audio_video_conflict_ratio"] = round(conflict_ratio, 3)

    if not reasons:
        reasons.append("Segnali misti o neutri")

    return {
        "result": {
            "label": label,
            "ai_score": float(round(ai_score, 6)),
            "confidence": int(confidence),
            "reason": "; ".join(reasons),
        },
        "timeline_binned": timeline_binned,
        "peaks": peaks,
        "hints": hints,
    }
