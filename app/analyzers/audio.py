# app/analyzers/audio.py
import os
import tempfile
import subprocess
from typing import Dict, Any, List

import numpy as np
import soundfile as sf
import librosa

def _run(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)

def _extract_wav(path: str, sr: int = 16000) -> str:
    tmp = tempfile.mkdtemp(prefix="aud_")
    out = os.path.join(tmp, "audio.wav")
    # mono, target sr
    _run(["ffmpeg", "-hide_banner", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), out])
    return out

def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    mn = float(np.min(x))
    ptp = float(np.ptp(x))
    if ptp == 0.0:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / ptp

def analyze(path: str, target_sr: int = 16000) -> Dict[str, Any]:
    """
    Output:
      "scores": {"audio_mean": float, "tts_like": float, "hnr_proxy": float},
      "flags_audio": [str, ...],
      "timeline": [{"start":s, "end":e, "ai_score":float}, ...]
    NOTE: `target_sr` è accettato (compatibile con api.py) e passato all’estrazione WAV.
    """
    wav = _extract_wav(path, sr=target_sr)
    try:
        y, sr = sf.read(wav)  # già mono
        flags: List[str] = []

        if y is None or len(y) == 0:
            return {"scores": {"audio_mean": 0.5, "tts_like": 0.0, "hnr_proxy": 0.0},
                    "flags_audio": ["no_audio"], "timeline": []}
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = np.mean(y, axis=1)
        y = np.asarray(y, dtype=np.float32)

        duration = float(len(y)) / float(sr) if sr > 0 else 0.0
        if duration <= 0.0:
            return {"scores": {"audio_mean": 0.5, "tts_like": 0.0, "hnr_proxy": 0.0},
                    "flags_audio": ["no_audio"], "timeline": []}

        # ---- Finestratura coerente: 1 secondo, stesse dimensioni per tutte le feature ----
        win = int(sr)      # frame di 1s
        hop = int(sr)      # hop di 1s
        n_fft = win        # n_fft uguale a frame_length per evitare lunghezze diverse
        win_len = win

        # Calcolo feature (center=False per allineare bene gli indici)
        rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=False)[0]  # shape ~ n_sec
        flat = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop,
                                                 win_length=win_len, center=False)[0]
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=win, hop_length=hop, center=False)[0]

        # ---- Allineamento lunghezze (usa la minima) ----
        L = int(min(rms.shape[0], flat.shape[0], zcr.shape[0]))
        if L <= 0:
            return {"scores": {"audio_mean": 0.5, "tts_like": 0.0, "hnr_proxy": 0.0},
                    "flags_audio": ["no_audio"], "timeline": []}

        rms = rms[:L]
        flat = flat[:L]
        zcr = zcr[:L]

        # Normalizzazioni
        rms_n  = _norm(rms)
        flat_n = _norm(flat)
        zcr_n  = _norm(zcr)

        # Variabilità globale come proxy prosodico
        rms_std = float(np.std(rms_n)) if rms_n.size else 0.0
        zcr_std = float(np.std(zcr_n)) if zcr_n.size else 0.0

        # Proxy HNR rozzo (1 - flatness media normalizzata)
        hnr_proxy = float(1.0 - (float(np.mean(flat_n)) if flat_n.size else 0.0))

        # ---- Heuristica per-secondo (evita broadcasting errori) ----
        ai_per_sec = (
            0.40 * flat_n +
            0.35 * (1.0 - rms_n) +
            0.15 * (1.0 - min(1.0, max(0.0, rms_std))) +
            0.10 * (1.0 - min(1.0, max(0.0, zcr_std)))
        )

        # Flag semplici
        energy_mean   = float(np.mean(rms)) if rms.size else 0.0
        silence_ratio = float(np.mean(rms < (np.median(rms) * 0.3 + 1e-9))) if rms.size else 1.0
        if silence_ratio > 0.7:
            flags.append("mostly_silent")
        if energy_mean < 1e-3:
            flags.append("very_low_energy")

        # ---- Costruzione timeline per-secondo ----
        # Numero di secondi “attesi” in base alla durata reale
        n_bins = int(np.ceil(duration))
        n_bins = max(1, min(n_bins, 180))  # limite di sicurezza 3 minuti

        # Se le feature hanno meno frame di n_bins, fai padding replicando l'ultimo valore
        per_frames = ai_per_sec.tolist()
        if len(per_frames) == 0:
            per_frames = [0.5]
        if len(per_frames) < n_bins:
            last = per_frames[-1]
            per_frames = per_frames + [last] * (n_bins - len(per_frames))
        else:
            per_frames = per_frames[:n_bins]

        timeline: List[Dict[str, float]] = [
            {"start": float(i), "end": float(i + 1), "ai_score": float(per_frames[i])}
            for i in range(n_bins)
        ]

        # Warm-up: neutralizza il primo secondo
        if len(timeline) >= 1:
            timeline[0]["ai_score"] = 0.5

        # Smoothing 3-punti
        if len(timeline) >= 3:
            sm = []
            for i in range(len(timeline)):
                lo = max(0, i-1); hi = min(len(timeline)-1, i+1)
                sm.append({"start": timeline[i]["start"], "end": timeline[i]["end"],
                           "ai_score": float(np.mean([timeline[j]["ai_score"] for j in range(lo,hi+1)]))})
            timeline = sm

        # Flags più severi
        if float(np.mean(flat_n)) >= 0.55 if flat_n.size else False:
            flags.append("tts_like")
        if float(np.mean(rms_n)) <= 0.05 if rms_n.size else False:
            flags.append("very_low_energy")

        return {
            "scores": {
                "audio_mean": float(np.mean([t["ai_score"] for t in timeline])) if timeline else 0.5,
                "tts_like": float(np.mean(flat_n)) if flat_n.size else 0.0,
                "hnr_proxy": hnr_proxy,
            },
            "flags_audio": flags,
            "timeline": timeline
        }
    finally:
        try:
            os.remove(wav)
            os.rmdir(os.path.dirname(wav))
        except Exception:
            pass