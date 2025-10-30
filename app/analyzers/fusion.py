import os
import numpy as np

THRESH_REAL_MAX = float(os.getenv("THRESH_REAL_MAX", "0.35"))
THRESH_AI_MIN   = float(os.getenv("THRESH_AI_MIN", "0.72"))

def _bin_timeline(ts):
    if not ts:
        return []
    arr = np.array(ts, dtype=float)
    if len(arr) >= 3:
        ker = np.ones(3)/3.0
        arr = np.convolve(arr, ker, mode="same")
    return np.clip(arr, 0.0, 1.0).tolist()

def fuse(audio: dict, video: dict, hints: dict):
    a_t = audio.get("timeline") or []
    v_t = video.get("timeline") or video.get("timeline_ai") or []
    L = max(len(a_t), len(v_t), 1)
    if len(a_t) < L: a_t += [a_t[-1] if a_t else 0.5]*(L-len(a_t))
    if len(v_t) < L: v_t += [v_t[-1] if v_t else 0.5]*(L-len(v_t))

    a = np.array(a_t, dtype=float)
    v = np.array(v_t, dtype=float)

    # Pesi base conservativi
    w_audio = 0.65
    w_video = 0.25
    bonus_agree = 0.10 if np.sign(np.mean(a)-0.5) == np.sign(np.mean(v)-0.5) else 0.0

    # Dinamica pesi dal parlato
    flags = audio.get("flags_audio", {})
    speech_ratio = float(flags.get("speech_ratio", 0.0))
    tts_like = float(flags.get("tts_like", 0.0))
    if speech_ratio < 0.25:
        w_audio *= 0.6
        w_video = max(0.2, 1.0 - w_audio - bonus_agree)

    # Penalità qualità/compressione/duplicati
    comp = hints.get("compression", "normal")
    bpp  = hints.get("bpp", 0.0)
    dup  = hints.get("dup_avg", 0.0)
    penalties = 0.0
    if comp in ("heavy", "very_heavy"): penalties += 0.05
    if bpp < 0.07: penalties += 0.05
    if dup > 0.2: penalties += 0.05

    # Bonus “ripresa reale”
    vsum = video.get("summary", {}) or {}
    flow_mean = float(vsum.get("flow_mean", 0.0))
    texture_var = float(vsum.get("texture_var", 0.0))
    sc_rate = float(vsum.get("scene_change_rate", 0.0))
    dup_density = float(vsum.get("dup_density", 0.0))

    real_bonus = 0.0
    if flow_mean > 5.0 and texture_var > 200.0 and dup_density < 0.05:
        real_bonus -= 0.10
    if sc_rate > 0.7:
        real_bonus -= 0.05
    if sc_rate >= 0.9 and texture_var > 300.0 and dup_density < 0.02:
        real_bonus -= 0.08

    # Se tts_like molto alto ma video fortemente reale → smorza ancora l'audio
    if tts_like >= 0.95 and flow_mean > 8.0 and texture_var > 300.0 and dup_density < 0.05:
        w_audio *= 0.55
        w_video = max(0.25, 1.0 - w_audio - bonus_agree)

    # Combinazione
    timeline = (w_audio*a + w_video*v + bonus_agree*(a+v)/2.0) - penalties + real_bonus
    timeline = np.clip(timeline, 0.0, 1.0)

    # Picchi (escludi ~0.5)
    peaks = [i for i, x in enumerate(timeline.tolist()) if x <= 0.25 or x >= 0.75]

    score = float(np.mean(timeline))
    spread = float(np.std(timeline))
    disagree = float(abs(np.mean(a) - np.mean(v)))

    conf = float(np.clip(0.20 + 2.2*spread - penalties - 0.5*max(0.0, 0.3 - disagree), 0.10, 0.99))

    if score <= THRESH_REAL_MAX:
        label = "real"
        reason = []
        if dup_density > 0.25: reason.append("molti frame duplicati")
        if comp in ("heavy","very_heavy"): reason.append("compressione pesante")
        if not reason: reason.append("segnali audio/video coerenti con ripresa reale")
        reason = "; ".join(reason)
    elif score >= THRESH_AI_MIN:
        label = "ai"
        reason = []
        if tts_like > 0.6: reason.append("audio TTS-like elevato")
        if dup_density > 0.2: reason.append("molti frame duplicati")
        if hints.get("video_has_signal", True) is False: reason.append("segnali video deboli")
        if not reason: reason = ["pattern e indizi coerenti con generazione AI"]
        reason = "; ".join(reason)
    else:
        label = "uncertain"
        reason = "segnali misti o neutri"

    return {
        "result": {
            "label": label,
            "ai_score": round(score, 2),
            "confidence": round(conf, 2),
            "reason": reason
        },
        "timeline_binned": _bin_timeline(timeline.tolist()),
        "peaks": peaks,
    }