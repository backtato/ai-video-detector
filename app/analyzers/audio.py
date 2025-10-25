# app/analyzers/audio.py
import os
import json
import tempfile
import subprocess
from typing import Dict, Any, List

import numpy as np
import librosa
import soundfile as sf

def _run(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)

def _extract_wav(path: str, sr: int = 16000) -> str:
    """
    Estrae una traccia WAV mono a 16 kHz in una cartella temporanea e ritorna il path.
    """
    tmp = tempfile.mkdtemp(prefix="aud_")
    out = os.path.join(tmp, "audio.wav")
    cmd = [
        "ffmpeg","-hide_banner","-v","error",
        "-i", path,
        "-ac","1","-ar", str(sr),
        out
    ]
    _run(cmd)
    return out

def analyze_audio(path: str, sr: int = 16000) -> Dict[str, Any]:
    """
    Analisi audio con librosa: energia media, var. del centroid, pause ratio.
    Ritorna:
      - scores.audio_mean
      - flags_audio[]
      - timeline_audio (bins grossolani di 1 s con “ai_score” derivato)
    """
    wav = _extract_wav(path, sr=sr)
    flags: List[str] = []
    try:
        y, sr_ = librosa.load(wav, sr=sr, mono=True)
        if y.size == 0:
            return {
                "scores": {"audio_mean": 0.5},
                "flags_audio": ["no_audio_data"],
                "timeline": [{"start":0.0,"end":1.0,"ai_score":0.5}]
            }

        # Energia RMS frame-based
        hop = 512
        frame_rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop).flatten()
        energy_mean = float(np.mean(frame_rms))
        energy_std = float(np.std(frame_rms))

        # Pause ratio: percentuale di frame con RMS molto basso
        thr = 0.2 * (np.median(frame_rms) + 1e-8)
        silence_ratio = float(np.mean(frame_rms < thr))

        # Spettrali: spectral centroid (varianza)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop).flatten()
        cent_var = float(np.var(cent / (sr/2.0) if (sr > 0) else cent))

        # Score audio [0..1]: più flat/robotico → bias AI
        # Mix sobrio per non sovrappesare l’audio
        score = float(np.clip(0.4*(1.0 - silence_ratio) + 0.3*np.clip(cent_var,0,1) + 0.3*np.clip(energy_std/(energy_mean+1e-8),0,1), 0.0, 1.0))

        # Hint
        if silence_ratio > 0.5:
            flags.append("long_pauses")
        if energy_mean < 0.005:
            flags.append("very_low_energy")

        # Timeline audio (1 s bins) — map semplificata dei frame in secondi
        duration = len(y)/sr if sr>0 else 0.0
        n_bins = max(int(duration), 1)
        if n_bins > 180:  # limita per output
            n_bins = 180
        timeline = []
        for i in range(n_bins):
            timeline.append({"start": float(i), "end": float(i+1), "ai_score": float(score)})

        return {
            "scores": {"audio_mean": score},
            "flags_audio": flags,
            "timeline": timeline
        }
    finally:
        try:
            os.remove(wav)
            os.rmdir(os.path.dirname(wav))
        except Exception:
            pass