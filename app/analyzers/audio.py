# app/analyzers/audio.py
import os, tempfile, subprocess
from typing import Dict, Any, List
import numpy as np
import librosa
import soundfile as sf

def _run(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)

def _extract_wav(path: str, sr: int = 16000) -> str:
    tmp = tempfile.mkdtemp(prefix="aud_")
    out = os.path.join(tmp, "audio.wav")
    _run(["ffmpeg","-hide_banner","-v","error","-i", path, "-ac","1","-ar",str(sr), out])
    return out

def analyze_audio(path: str, sr: int=16000) -> Dict[str, Any]:
    wav = _extract_wav(path, sr=sr)
    flags: List[str] = []
    try:
        y, sr_ = librosa.load(wav, sr=sr, mono=True)
        if y.size == 0:
            return {"scores": {"audio_mean": 0.5}, "flags_audio": ["no_audio_data"],
                    "timeline": [{"start":0.0,"end":1.0,"ai_score":0.5}]}

        hop = 512
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop).flatten()
        energy_mean, energy_std = float(np.mean(rms)), float(np.std(rms))
        thr = 0.2*(np.median(rms)+1e-8)
        silence_ratio = float(np.mean(rms < thr))

        cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop).flatten()
        cent_var = float(np.var(cent/(sr/2.0) if sr>0 else cent))

        score = float(np.clip(
            0.4*(1.0 - silence_ratio) +
            0.3*np.clip(cent_var,0,1) +
            0.3*np.clip(energy_std/(energy_mean+1e-8),0,1),
            0.0, 1.0
        ))

        if silence_ratio > 0.5: flags.append("long_pauses")
        if energy_mean < 0.005: flags.append("very_low_energy")

        duration = len(y)/sr if sr>0 else 0.0
        n_bins = max(int(duration), 1)
        if n_bins>180: n_bins = 180
        timeline = [{"start": float(i), "end": float(i+1), "ai_score": float(score)} for i in range(n_bins)]

        return {"scores": {"audio_mean": score}, "flags_audio": flags, "timeline": timeline}
    finally:
        try:
            os.remove(wav)
            os.rmdir(os.path.dirname(wav))
        except: pass
