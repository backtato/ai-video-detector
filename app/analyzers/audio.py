# app/analyzers/audio.py
import os
import tempfile
import subprocess
from typing import Dict, Any, List, Optional

import numpy as np
import soundfile as sf
import librosa

def _run(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)

def _extract_wav(path: str, sr: int = 16000) -> str:
    tmp = tempfile.mkdtemp(prefix="aud_")
    out = os.path.join(tmp, "audio.wav")
    # mono, target sr
    _run(["ffmpeg","-hide_banner","-v","error","-i", path, "-ac","1","-ar",str(sr), out])
    return out

def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    mn = np.min(x)
    ptp = np.ptp(x)
    if ptp == 0:
        return np.zeros_like(x)
    return (x - mn) / ptp

def analyze(path: str, target_sr: int = 16000) -> Dict[str, Any]:
    """
    Estrae feature audio e produce una timeline *per secondo*.
    - Converte a WAV mono @target_sr
    - Feature: RMS (energia), spectral flatness, ZCR
    - Euristica *placeholder* per TTS/AI-like per-secondo
    Restituisce:
    {
      "scores": {"audio_mean": float, "tts_like": float, "hnr_proxy": float},
      "flags_audio": [...],
      "timeline": [{"start":s, "end":e, "ai_score":float}, ...]
    }
    """
    wav = _extract_wav(path, sr=target_sr)
    flags: List[str] = []
    try:
        # Carica segnale
        y, sr = sf.read(wav, always_2d=False)
        if y is None or (isinstance(y, np.ndarray) and y.size == 0):
            return {"scores": {"audio_mean": 0.5, "tts_like": 0.0, "hnr_proxy": 0.0}, "flags_audio": ["no_audio"], "timeline": []}
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = np.mean(y, axis=1)
        y = np.asarray(y, dtype=np.float32)

        duration = float(len(y)) / float(sr) if sr > 0 else 0.0
        if duration <= 0.0:
            return {"scores": {"audio_mean": 0.5, "tts_like": 0.0, "hnr_proxy": 0.0}, "flags_audio": ["no_audio"], "timeline": []}

        # Parametri finestra/hop per timeline 1s
        win = sr
        hop = sr

        # Feature per frame-secondo
        rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=False)[0]
        # Evita FFT più grande del segnale
        n_fft = 2048 if sr >= 2048 else max(512, int(sr/2))
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop, win_length=min(n_fft, win), center=False)[0]
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=win, hop_length=hop, center=False)[0]

        # Normalizzazioni e statistiche
        rms_n  = _norm(rms)
        flat_n = _norm(flatness)
        zcr_n  = _norm(zcr)

        # Variabilità (prosodia/intonazione proxy) – std globali normalizzati
        rms_std = float(np.std(rms_n)) if rms_n.size else 0.0
        zcr_std = float(np.std(zcr_n)) if zcr_n.size else 0.0

        # Proxy HNR rozzo (1 - flatness media normalizzata)
        hnr_proxy = float(1.0 - float(np.mean(flat_n)) if flat_n.size else 0.0)

        # Heuristica TTS/AI-like per secondo:
        # maggiore flatness + bassa variabilità/energia → più “AI-like”
        ai_per_sec = 0.4*flat_n + 0.35*(1.0 - rms_n) + 0.15*(1.0 - rms_std) + 0.10*(1.0 - zcr_std)
        ai_per_sec = np.clip(ai_per_sec, 0.0, 1.0)

        # Flag semplici
        energy_mean    = float(np.mean(rms)) if rms.size else 0.0
        silence_ratio  = float(np.mean(rms < (np.median(rms)*0.3 + 1e-9))) if rms.size else 1.0
        if silence_ratio > 0.7:
            flags.append("mostly_silent")
        if energy_mean < 1e-3:
            flags.append("very_low_energy")

        # Timeline per-secondo (limite di sicurezza 180s)
        n_bins = int(np.ceil(duration))
        n_bins = max(1, min(n_bins, 180))
        timeline: List[Dict[str, float]] = []
        for i in range(n_bins):
            score = float(ai_per_sec[i]) if i < ai_per_sec.size else float(np.mean(ai_per_sec)) if ai_per_sec.size else 0.5
            timeline.append({"start": float(i), "end": float(i+1), "ai_score": score})

        return {
            "scores": {
                "audio_mean": float(np.mean(ai_per_sec)) if ai_per_sec.size else 0.5,
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
