import numpy as np

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _mean(xs):
    return float(sum(xs) / max(1, len(xs)))

def _voice_presence(rms_win):
    if not rms_win:
        return 0.0
    thr = np.percentile(rms_win, 60)
    active = sum(1 for r in rms_win if r >= thr)
    return active / len(rms_win)

def analyze(wav_16k_mono: np.ndarray, sr: int, duration_sec: float) -> dict:
    if wav_16k_mono is None or wav_16k_mono.size == 0 or sr <= 0:
        return {
            "scores": {"audio_mean": 0.5, "tts_like": 0.5, "hnr_proxy": 0.5},
            "flags_audio": ["low_voice_presence"],
            "timeline": []
        }

    win = sr  # 1s
    n = len(wav_16k_mono)
    nwin = max(1, int(np.ceil(n / win)))
    timeline = []
    rms_list = []
    zcr_list = []
    roll_list = []
    flat_list = []

    for i in range(nwin):
        s = i * win
        e = min(n, (i + 1) * win)
        seg = wav_16k_mono[s:e].astype(np.float32)
        if seg.size == 0:
            timeline.append({"start": i, "end": min(i + 1, duration_sec), "ai_score": 0.5})
            continue

        rms = float(np.sqrt(np.mean(seg ** 2)))
        rms_list.append(rms)

        zc = np.mean(np.abs(np.diff(np.sign(seg)))) if seg.size > 1 else 0.0
        zcr_list.append(float(zc))

        fft = np.abs(np.fft.rfft(seg, n=2048))
        cumsum = np.cumsum(fft)
        thr = 0.85 * cumsum[-1] if cumsum[-1] > 0 else 0
        roll_idx = int(np.searchsorted(cumsum, thr))
        rolloff = roll_idx / 1024.0
        roll_list.append(float(rolloff))

        flat = float(np.std(fft) / (np.mean(fft) + 1e-8)) if fft.size > 0 else 0.0
        flat_list.append(flat)

    vad = _voice_presence(rms_list)

    flags = []
    if vad < 0.25:
        flags.append("low_voice_presence")

    likely_music = (_mean(roll_list) > 0.45 and _mean(flat_list) < 0.9 and _mean(zcr_list) < 0.25)
    if likely_music:
        flags.append("likely_music")

    timeline_scores = []
    for i in range(nwin):
        base = 0.48 + 0.1 * (0.5 - min(flat_list[i], 1.0)) + 0.05 * (roll_list[i] - 0.4)
        ai = float(base)

        if "low_voice_presence" in flags:
            ai = min(ai, 0.60)
        if likely_music:
            ai = min(ai, 0.60)

        ai = _clamp(ai, 0.35, 0.70)
        timeline_scores.append(ai)
        timeline.append({"start": i, "end": min(i + 1, duration_sec), "ai_score": ai})

    out = {
        "scores": {
            "audio_mean": _mean(timeline_scores) if timeline_scores else 0.5,
            "tts_like": _mean(flat_list) if flat_list else 0.5,
            "hnr_proxy": 1.0 - (_mean(zcr_list) if zcr_list else 0.5)
        },
        "flags_audio": flags,
        "timeline": timeline
    }
    return out
