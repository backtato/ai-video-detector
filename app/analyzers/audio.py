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
    cmd = [
        "ffmpeg", "-y", "-i", path,
        "-ac", "1", "-ar", str(sr),
        "-map_metadata", "-1", "-vn", "-sn", "-dn", out
    ]
    _run(cmd)
    return out

def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    mn = float(np.min(x)); ptp = float(np.ptp(x))
    if ptp == 0.0:
        return np.zeros_like(x)
    return (x - mn) / ptp

def analyze(path: str, target_sr: int = 16000) -> Dict[str, Any]:
    """
    Timeline per-secondo con features semplici + VAD leggero.
    In assenza di voce, limita l'impatto dei pattern "tts-like".
    """
    wav = _extract_wav(path, sr=target_sr)
    flags: List[str] = []
    try:
        y, sr = sf.read(wav, always_2d=False)
        if y is None:
            return {"scores": {"audio_mean": 0.5, "tts_like": 0.0, "hnr_proxy": 0.0},
                    "flags_audio": ["no_audio"], "timeline": []}
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = np.mean(y, axis=1)
        y = np.asarray(y, dtype=np.float32)

        duration = float(len(y)) / float(sr) if sr > 0 else 0.0
        if duration <= 0.0:
            return {"scores": {"audio_mean": 0.5, "tts_like": 0.0, "hnr_proxy": 0.0},
                    "flags_audio": ["no_audio"], "timeline": []}

        win = int(sr)      # 1s
        hop = int(sr)
        n_fft = win
        win_len = win

        rms  = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=False)[0]
        flat = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop, win_length=win_len, center=False)[0]
        zcr  = librosa.feature.zero_crossing_rate(y, frame_length=win, hop_length=hop, center=False)[0]
        roll = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop, roll_percent=0.85, n_fft=n_fft, center=False)[0]

        L = int(min(rms.shape[0], flat.shape[0], zcr.shape[0], roll.shape[0]))
        if L <= 0:
            return {"scores": {"audio_mean": 0.5, "tts_like": 0.0, "hnr_proxy": 0.0},
                    "flags_audio": ["no_audio"], "timeline": []}

        rms, flat, zcr, roll = rms[:L], flat[:L], zcr[:L], roll[:L]

        rms_n   = _norm(rms)
        flat_n  = _norm(flat)
        zcr_n   = _norm(zcr)
        roll_n  = _norm(roll)

        # VAD grezzo: voce se energia > mediana e rolloff non troppo alto
        energy_med = float(np.median(rms))
        vad = (rms > (energy_med * 1.15)) & (roll_n < 0.75)
        vad_ratio = float(np.mean(vad)) if vad.size else 0.0

        # HNR proxy: dinamica energia vs flatness
        hnr_proxy = float(np.clip((np.std(rms_n) + (1.0 - np.mean(flat_n))) / 2.0, 0.0, 1.0))

        # AI per-secondo (euristica conservativa)
        ai_per_sec = (
            0.35 * flat_n +
            0.30 * (1.0 - rms_n) +
            0.20 * (1.0 - np.clip(np.std(rms_n), 0.0, 1.0)) +
            0.15 * (1.0 - np.clip(np.std(zcr_n), 0.0, 1.0))
        ).astype(float)

        # In assenza di voce, cap per evitare falsi positivi da rumore/NR
        if vad_ratio < 0.25:
            flags.append("low_voice_presence")
            ai_per_sec = np.minimum(ai_per_sec, 0.60)

        ai_per_sec = np.clip(ai_per_sec, 0.0, 1.0)

        energy_mean   = float(np.mean(rms)) if rms.size else 0.0
        silence_ratio = float(np.mean(rms < (energy_med * 0.3 + 1e-9))) if rms.size else 1.0
        if silence_ratio > 0.7:
            flags.append("mostly_silent")
        if energy_mean < 1e-3:
            flags.append("very_low_energy")

        # Timeline per-secondo
        n_bins = int(np.ceil(duration))
        n_bins = max(1, min(n_bins, 180))
        per_frames = ai_per_sec.tolist() or [0.5]
        if len(per_frames) < n_bins:
            per_frames += [per_frames[-1]] * (n_bins - len(per_frames))

        timeline = [{"start": i, "end": i + 1, "ai_score": float(np.clip(per_frames[i], 0.0, 1.0))}
                    for i in range(n_bins)]

        return {
            "scores": {
                "audio_mean": float(np.mean(per_frames)) if per_frames else 0.5,
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
