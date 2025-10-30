import numpy as np

def _clamp(v, lo, hi): return max(lo, min(hi, v))
def _mean(xs): return float(sum(xs) / max(1, len(xs)))

def _voice_presence(rms_win):
    if not rms_win: return 0.0
    import numpy as np
    thr = np.percentile(rms_win, 60)
    active = sum(1 for r in rms_win if r >= thr)
    return active / len(rms_win)

def analyze(wav_16k_mono: np.ndarray, sr: int, duration_sec: float) -> dict:
    if wav_16k_mono is None or sr <= 0 or duration_sec <= 0:
        return {"scores": {"audio_mean": 0.5}, "flags_audio": ["no_audio"], "timeline": []}

    win = int(sr * 0.5); hop = win
    n = len(wav_16k_mono); nwin = max(1, n // hop)

    rms_list, zcr_list, roll_list, flat_list = [], [], [], []

    for i in range(nwin):
        s = i * hop; e = min(n, s + win)
        seg = wav_16k_mono[s:e].astype(np.float32)
        if len(seg) == 0: continue
        seg = seg - np.mean(seg)

        rms = float(np.sqrt(np.mean(seg * seg) + 1e-9)); rms_list.append(rms)
        zc = np.mean(np.abs(np.diff(np.signbit(seg)).astype(np.float32))); zcr_list.append(float(zc))
        thr = float(np.percentile(np.abs(seg), 90)); roll = float(np.mean(np.abs(seg) > (0.5 * thr))); roll_list.append(roll)
        mu = float(np.mean(np.abs(seg)) + 1e-8); va = float(np.var(seg)); flat = _clamp(va / (mu * mu + 1e-8), 0.0, 2.0); flat_list.append(flat)

    vad = _voice_presence(rms_list)
    flags = []
    if vad < 0.25: flags.append("low_voice_presence")
    likely_music = (_mean(roll_list) > 0.45 and _mean(flat_list) < 0.9 and _mean(zcr_list) < 0.25)
    if likely_music: flags.append("likely_music")

    timeline_scores = []
    for i in range(nwin):
        base = 0.5
        base -= 0.15 * _clamp(vad, 0.0, 1.0)
        base += 0.10 * _clamp(_mean(zcr_list[i:i+1]), 0.0, 1.0)
        base -= 0.05 * _clamp(_mean(roll_list[i:i+1]), 0.0, 1.0)
        timeline_scores.append(_clamp(base, 0.0, 1.0))

    audio_mean = float(_mean(timeline_scores)) if timeline_scores else 0.5
    return {"scores": {"audio_mean": audio_mean}, "flags_audio": flags, "timeline": timeline_scores}
