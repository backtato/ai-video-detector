# app/fusion.py
# Fusione adattiva video+audio+hints; soglie dinamiche per bitrate; confidenza leggibile;
# reason codes più espliciti. Mantiene chiave 'result' e campi timeline_binned/peaks.

import numpy as np

BASE_THRESH_REAL = 0.38
BASE_THRESH_AI   = 0.72
W_VIDEO = 0.55
W_AUDIO = 0.35
W_HINTS = 0.10

def _score_from_video(vstats: dict):
    if not vstats or "timeline" not in vstats:
        return 0.5, []
    tl = vstats.get("timeline", [])
    motion = np.array([x.get("motion", 0.0) for x in tl], dtype=np.float32)
    edge   = np.array([x.get("edge_var", 0.0) for x in tl], dtype=np.float32)
    dup    = np.array([x.get("dup", 0.0) for x in tl], dtype=np.float32)
    block  = np.array([x.get("blockiness", 0.0) for x in tl], dtype=np.float32)
    band   = np.array([x.get("banding", 0.0) for x in tl], dtype=np.float32)

    def rn(x):
        if x.size == 0: return x
        xm, xs = float(x.mean()), float(x.std() or 1.0)
        z = (x - xm) / xs
        return 1.0 / (1.0 + np.exp(-z))

    # Heuristica: più dup/block/band → più probabile sintetico/schermo
    v_ai = np.clip(0.25*(1.0 - rn(motion)) +
                   0.25*(1.0 - rn(edge)) +
                   0.25*rn(dup) +
                   0.25*(rn(block) + rn(band))/2.0, 0.0, 1.0)
    score = float(v_ai.mean()) if v_ai.size else 0.5
    return score, v_ai.tolist()

def _score_from_audio(astats: dict):
    if not astats: return 0.5, []
    tts_like = float(astats.get("scores", {}).get("tts_like", 0.0))
    audio_mean = float(astats.get("scores", {}).get("audio_mean", 0.5))
    score = float(0.6*audio_mean + 0.4*tts_like)
    timeline = astats.get("timeline", [])
    return score, [float(x.get("ai_score", score)) for x in timeline]

def _score_from_hints(hints: dict):
    if not hints: return 0.5
    vals = [float(v.get("score", 0.5)) for v in hints.values() if isinstance(v, dict)]
    return float(np.mean(vals)) if vals else 0.5

def _bin_by_second(seq, duration):
    if not isinstance(seq, (list, tuple)) or not seq:
        return []
    n = min(int(duration) if duration else len(seq), len(seq))
    out = []
    for i in range(n):
        out.append({"start": float(i), "end": float(i+1), "ai_score": float(seq[i])})
    return out

def fuse(video_stats=None, audio_stats=None, hints=None, meta=None):
    hints = hints or {}
    meta = meta or {}
    duration = 0.0
    try:
        duration = float(video_stats.get("duration", 0.0))
    except Exception:
        pass

    v_score, v_seq = _score_from_video(video_stats)
    a_score, a_seq = _score_from_audio(audio_stats)
    h_score = _score_from_hints(hints)

    # Pesi adattivi se audio manca
    wv, wa, wh = W_VIDEO, W_AUDIO, W_HINTS
    if not audio_stats or not a_seq:
        wa = 0.0
        total = wv + wh
        if total > 0:
            wv, wh = wv/total, wh/total

    ai_score = float(np.clip(wv * v_score + wa * a_score + wh * h_score, 0.0, 1.0))

    # Soglie dinamiche per bitrate basso (WhatsApp/schermo)
    bit_rate = float(meta.get("bit_rate", 0.0) or meta.get("summary", {}).get("bit_rate", 0.0) or 0.0)
    t_real, t_ai = BASE_THRESH_REAL, BASE_THRESH_AI
    if bit_rate and bit_rate < 800_000:
        t_real = 0.34
        t_ai   = 0.76

    # Label
    if ai_score >= t_ai:
        label = "ai"
    elif ai_score <= t_real:
        label = "real"
    else:
        label = "uncertain"

    # Confidenza (0..1) → UI la può mappare a percentuale
    dist = max(ai_score - t_ai, t_real - ai_score, 0.0)
    span = max(t_ai - t_real, 1e-6)
    confidence = float(np.clip(0.5 + (dist / span), 0.0, 1.0))

    # Timeline binned (merge semplice video/audio)
    max_len = int(max(len(v_seq), len(a_seq)))
    per_sec = []
    for i in range(max_len):
        vs = v_seq[i] if i < len(v_seq) else None
        as_ = a_seq[i] if i < len(a_seq) else None
        if vs is None and as_ is None:
            continue
        if vs is None: per_sec.append(as_)
        elif as_ is None: per_sec.append(vs)
        else: per_sec.append(float(0.6*vs + 0.4*as_))
    timeline_binned = _bin_by_second(per_sec, duration)

    # Peaks
    peaks = [{"t": float(i), "ai_score": float(s)} for i, s in enumerate(per_sec) if s is not None and s >= 0.85][:10]

    # Reason codes
    reasons = []
    neg = [k for k, v in (hints or {}).items() if isinstance(v, dict) and v.get("score", 0.5) >= 0.6][:5]
    pos = [k for k, v in (hints or {}).items() if isinstance(v, dict) and v.get("score", 0.5) <= 0.45][:5]
    if neg: reasons.append("Hints−: " + ", ".join(neg))
    if pos: reasons.append("Hints+: " + ", ".join(pos))
    if bit_rate and bit_rate < 800_000: reasons.append("Bitrate molto basso → soglie conservative.")
    if not a_seq: reasons.append("Audio assente o non utilizzabile.")
    if label == "uncertain": reasons.append("Segnali misti o qualità/compressione limitante.")

    return {
        "result": {
            "label": label,
            "ai_score": ai_score,
            "confidence": confidence,
            "reason": " ".join(reasons).strip()
        },
        "timeline_binned": timeline_binned,
        "peaks": peaks
    }
