import math
import numpy as np

THRESH_REAL = 0.38
THRESH_AI = 0.72

# pesi conservativi
W_VIDEO = 0.55
W_AUDIO = 0.35
W_HINTS = 0.10

def _score_from_video(vstats: dict) -> (float, list):
    if not vstats or "timeline" not in vstats:
        return 0.5, []

    tl = vstats.get("timeline", [])
    if not tl:
        return 0.5, []

    # Heuristica: più motion/edge ⇒ meno “sintetico”
    m = np.array([x.get("motion", 0.0) for x in tl], dtype=np.float32)
    e = np.array([x.get("edge_var", 0.0) for x in tl], dtype=np.float32)

    # Normalizzazioni robuste
    def robust_norm(x):
        if x.size == 0:
            return x
        xm = float(x.mean())
        xs = float(x.std()) or 1.0
        z = (x - xm) / xs
        # mappa z in [0,1]
        return 1.0 / (1.0 + np.exp(-z))

    m_n = robust_norm(m)  # 0..1
    e_n = robust_norm(e)  # 0..1

    # Se entrambe basse (~0) ⇒ potenziale “sintetico/schermo”, score AI ↑
    # Invertiamo il segno: 1 - max(m,e) => più basso dinamismo ⇒ score AI maggiore
    per_sec_ai = (1.0 - np.maximum(m_n, e_n)).clip(0.0, 1.0)
    score = float(per_sec_ai.mean()) if per_sec_ai.size else 0.5
    return score, per_sec_ai.tolist()

def _score_from_audio(astats: dict) -> (float, list):
    if not astats or "timeline" not in astats:
        return 0.5, []
    tl = astats["timeline"]
    # Semplice proxy: più silenzi/piattezza ⇒ più “sintetico”
    rms = np.array([x.get("rms", 0.0) for x in tl], dtype=np.float32)
    zcr = np.array([x.get("zcr", 0.0) for x in tl], dtype=np.float32)

    # Normalizza robustamente
    def rn(x):
        if x.size == 0:
            return x
        xm = float(x.mean())
        xs = float(x.std()) or 1.0
        z = (x - xm) / xs
        return 1.0 / (1.0 + np.exp(-z))

    rms_n = rn(rms)
    zcr_n = rn(zcr)

    # Se entrambe basse ⇒ score AI ↑
    per_sec_ai = (1.0 - np.maximum(rms_n, zcr_n)).clip(0.0, 1.0)
    score = float(per_sec_ai.mean()) if per_sec_ai.size else 0.5
    return score, per_sec_ai.tolist()

def _score_from_hints(hints: dict) -> float:
    if not hints:
        return 0.5
    # media pesata semplice: flag negativi spingono verso AI (↑)
    # normalizziamo in [0,1]
    vals = []
    for k, v in hints.items():
        if isinstance(v, dict) and "score" in v:
            vals.append(float(v.get("score", 0.5)))
    if not vals:
        return 0.5
    return float(np.mean(vals))

def _bin_by_second(per_sec_seq: list, duration: float) -> list:
    # per_sec_seq è già per-second in molti casi; qui lo allineiamo
    if not per_sec_seq:
        return []
    # taglia alla durata
    max_len = max(1, int(duration))
    arr = per_sec_seq[:max_len]
    return [{"t": i, "ai": float(arr[i])} for i in range(len(arr))]

def _find_peaks(bins: list, min_prom: float = 0.15) -> list:
    if not bins:
        return []
    arr = np.array([b["ai"] for b in bins], dtype=np.float32)
    peaks = []
    if arr.size < 3:
        return peaks
    mean = float(arr.mean())
    std = float(arr.std()) or 1.0
    for i in range(1, arr.size - 1):
        if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
            if (arr[i] - mean) > min_prom * std:
                peaks.append({"t": int(bins[i]["t"]), "ai": float(arr[i])})
    return peaks

def fuse(video_stats: dict, audio_stats: dict, hints: dict, meta: dict) -> dict:
    duration = 0.0
    try:
        if video_stats and video_stats.get("duration"):
            duration = float(video_stats["duration"])
        elif meta and "summary" in meta and meta["summary"].get("duration"):
            duration = float(meta["summary"]["duration"])
    except Exception:
        duration = 0.0

    v_score, v_seq = _score_from_video(video_stats)
    a_score, a_seq = _score_from_audio(audio_stats)
    h_score = _score_from_hints(hints)

    # Fusione conservativa
    ai_score = float(W_VIDEO * v_score + W_AUDIO * a_score + W_HINTS * h_score)
    ai_score = max(0.0, min(1.0, ai_score))

    # Label
    if ai_score >= THRESH_AI:
        label = "ai"
    elif ai_score <= THRESH_REAL:
        label = "real"
    else:
        label = "uncertain"

    # Timeline binned
    # Se abbiamo entrambe, facciamo media; altrimenti quella disponibile
    max_len = int(max(len(v_seq), len(a_seq)))
    per_sec = []
    for i in range(max_len):
        vs = v_seq[i] if i < len(v_seq) else None
        as_ = a_seq[i] if i < len(a_seq) else None
        if vs is None and as_ is None:
            continue
        if vs is None:
            per_sec.append(as_)
        elif as_ is None:
            per_sec.append(vs)
        else:
            per_sec.append((vs + as_) / 2.0)

    timeline_binned = _bin_by_second(per_sec, duration or len(per_sec))
    peaks = _find_peaks(timeline_binned)

    # Confidence = 1 - std (più stabile → più conf)
    if per_sec:
        std = float(np.std(np.array(per_sec, dtype=np.float32)))
        confidence = float(max(0.0, min(1.0, 1.0 - std)))
    else:
        confidence = 0.5

    # Reason sintetico
    reasons = []
    if label == "ai":
        reasons.append("Pattern di bassa dinamica/texture compatibili con contenuto sintetico o schermata.")
    elif label == "real":
        reasons.append("Dinamica e texture più coerenti con cattura reale.")
    else:
        reasons.append("Segnali contrastanti o insufficienti; analisi conservativa.")

    # Aggiungi hint principali
    if hints:
        neg = [k for k, v in hints.items() if isinstance(v, dict) and v.get("score", 0.5) > 0.6]
        if neg:
            reasons.append("Hints: " + ", ".join(neg))

    result = {
        "result": {
            "label": label,
            "ai_score": ai_score,
            "confidence": confidence,
            "reason": " ".join(reasons)
        },
        "timeline_binned": timeline_binned,
        "peaks": peaks
    }
    return result

