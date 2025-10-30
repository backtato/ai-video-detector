import os
import numpy as np

THRESH_REAL_MAX = float(os.getenv("THRESH_REAL_MAX", "0.35"))
THRESH_AI_MIN = float(os.getenv("THRESH_AI_MIN", "0.72"))

def _bin_timeline(ts):
    if not ts: return []
    # smooth with simple moving average (window=3)
    arr = np.array(ts, dtype=float)
    if len(arr)>=3:
        ker = np.ones(3)/3.0
        arr = np.convolve(arr, ker, mode="same")
    # bin per second already
    return np.clip(arr,0.0,1.0).tolist()

def fuse(audio: dict, video: dict, hints: dict):
    # audio timeline proxy: use audio.timeline if present, else neutral 0.5
    a_t = audio.get("timeline") or []
    v_t = video.get("timeline") or video.get("timeline_ai") or []

    L = max(len(a_t), len(v_t), 1)
    if len(a_t)<L: a_t += [a_t[-1] if a_t else 0.5]*(L-len(a_t))
    if len(v_t)<L: v_t += [v_t[-1] if v_t else 0.5]*(L-len(v_t))

    a = np.array(a_t, dtype=float)
    v = np.array(v_t, dtype=float)

    # base weights (conservative)
    w_audio = 0.65
    w_video = 0.25
    bonus_agree = 0.10 if np.sign(np.mean(a)-0.5) == np.sign(np.mean(v)-0.5) else 0.0

    # penalties/bonuses from hints
    comp = hints.get("compression","normal")
    bpp = hints.get("bpp", 0.0)
    dup = hints.get("dup_avg", 0.0)
    penalties = 0.0
    if comp in ("heavy","very_heavy"): penalties += 0.05
    if bpp < 0.07: penalties += 0.05
    if dup > 0.2: penalties += 0.05

    # combine
    timeline = (w_audio*a + w_video*v + bonus_agree*(a+v)/2.0) - penalties
    timeline = np.clip(timeline, 0.0, 1.0)

    # peaks (exclude near-0.5)
    peaks = [i for i,x in enumerate(timeline.tolist()) if x<=0.25 or x>=0.75]

    # final score as mean
    score = float(np.mean(timeline))
    # confidence from spread
    spread = float(np.std(timeline))
    conf = float(np.clip(0.20 + 2.5*spread - penalties, 0.10, 0.99))

    if score <= THRESH_REAL_MAX:
        label = "real"
        reason = []
        if dup>0.25: reason.append("molti frame duplicati")
        if comp in ("heavy","very_heavy"): reason.append("compressione pesante")
        if not reason: reason.append("segnali audio/video coerenti con ripresa reale")
        reason = "; ".join(reason)
    elif score >= THRESH_AI_MIN:
        label = "ai"
        reason = []
        if audio.get("scores",{}).get("tts_like",0)>0.6: reason.append("audio TTS-like elevato")
        if dup>0.2: reason.append("molti frame duplicati")
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