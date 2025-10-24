import wave, contextlib
import numpy as np
from typing import Tuple, List

def _spectral_flatness(x: np.ndarray, nfft: int=1024) -> float:
    if x.size == 0: return 0.5
    X = np.fft.rfft(x[:nfft] * np.hanning(min(len(x), nfft)))
    mag = np.abs(X) + 1e-9
    gm = np.exp(np.mean(np.log(mag)))
    am = np.mean(mag)
    return float(np.clip(gm / (am + 1e-9), 0.0, 1.0))

def _zero_crossing_rate(x: np.ndarray) -> float:
    if x.size <= 1: return 0.0
    return float(((x[:-1] * x[1:]) < 0).mean())

def audio_scores(wav_path: str, duration: float, window_sec: float=2.0) -> Tuple[List[float], List[dict]]:
    try:
        with contextlib.closing(wave.open(wav_path, 'rb')) as wf:
            sr = wf.getframerate()
            nframes = wf.getnframes()
            if nframes <= 0:
                return [0.5],[{"start":0.0,"end":window_sec,"ai_score":0.5}]
            data = wf.readframes(nframes)
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        return [0.5],[{"start":0.0,"end":window_sec,"ai_score":0.5}]

    hop = int(sr * window_sec)
    scores, timeline = [], []
    pos = 0
    t = 0.0
    while pos < len(x):
        seg = x[pos:pos+hop]
        if seg.size == 0: break
        zcr = _zero_crossing_rate(seg)
        flat = _spectral_flatness(seg)
        energy = float((seg**2).mean())
        zcr_n = np.clip(zcr / 0.2, 0.0, 1.0)
        flat_n = flat
        energy_n = np.clip(energy / 0.01, 0.0, 1.0)
        score = float(np.clip(0.5*flat_n + 0.3*zcr_n + 0.2*(1.0 - energy_n), 0.0, 1.0))
        scores.append(score)
        timeline.append({"start": float(t), "end": float(t+window_sec), "ai_score": score})
        pos += hop
        t += window_sec

    if not scores:
        scores = [0.5]
        timeline = [{"start":0.0,"end":window_sec,"ai_score":0.5}]
    return scores, timeline