# app/analyzers/audio.py
# Estensione: spectral flatness, MFCC variance, HNR proxy, timeline per secondo.
# Mantiene compatibilità: ritorna scores.audio_mean + timeline; aggiunge campi utili.

import os
import subprocess
import tempfile
import numpy as np
import librosa

def _extract_wav(in_path: str, target_sr: int = 16000) -> str:
    """Estrae un WAV mono 16k con ffmpeg. Ritorna path temporaneo."""
    tmpdir = tempfile.mkdtemp(prefix="aiv_audio_")
    out_wav = os.path.join(tmpdir, "audio.wav")
    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-ac", "1", "-ar", str(target_sr),
        "-vn", out_wav
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return out_wav

def analyze(path: str, target_sr: int = 16000) -> dict:
    wav = None
    try:
        wav = _extract_wav(path, target_sr=target_sr)
        if not os.path.exists(wav) or os.path.getsize(wav) == 0:
            return {"error": "audio_extract_failed"}

        # Caricamento librosa
        y, sr = librosa.load(wav, sr=target_sr, mono=True)
        if y.size == 0 or sr <= 0:
            return {"error": "audio_empty"}

        # Parametri finestra classici speech
        frame_length = int(0.025 * sr)  # 25 ms
        hop_length = int(0.010 * sr)    # 10 ms

        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        mfcc_var = np.var(mfcc, axis=1).mean()  # variabilità media MFCC

        # HNR proxy: energia banda bassa vs totale
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop_length)) + 1e-9
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        low_band = S[(freqs <= 3000), :]
        hnr_proxy = float(np.mean(low_band) / np.mean(S))

        # Normalizzazione robusta → sigmoide
        def rn(x):
            x = np.asarray(x, dtype=np.float32)
            if x.size == 0:
                return x
            xm, xs = float(x.mean()), float(x.std() or 1.0)
            z = (x - xm) / xs
            return 1.0 / (1.0 + np.exp(-z))

        rms_n = rn(rms)
        zcr_n = rn(zcr)
        flat_n = rn(flatness)
        cent_n = rn(centroid)

        # Euristica TTS/AI: bassa varianza MFCC + flatness alta + HNR basso
        tts_like = float(
            0.4 * (1.0 - np.tanh(mfcc_var / 50.0)) +
            0.3 * float(np.mean(flat_n)) +
            0.3 * (1.0 - np.tanh(hnr_proxy))
        )

        # Score complessivo audio (conservativo)
        per_sec_ai = np.clip(
            0.35 * (1.0 - np.maximum(rms_n, zcr_n)) +
            0.35 * tts_like +
            0.30 * (1.0 - np.tanh(np.std(cent_n))), 0.0, 1.0
        )
        score = float(np.mean(per_sec_ai)) if per_sec_ai.size else 0.5

        # Flags
        flags = []
        if float(np.mean(rms)) < 0.01:
            flags.append("very_low_energy")
        if float(np.mean(flatness)) > 0.35:
            flags.append("flat_spectrum")
        if tts_like > 0.6:
            flags.append("tts_like")

        # Timeline per secondo (fino a 180s max per coerenza)
        duration = len(y)/sr if sr > 0 else 0.0
        n_bins = max(int(duration), 1)
        if n_bins > 180:
            n_bins = 180
        timeline = [{"start": float(i), "end": float(i+1), "ai_score": float(score)} for i in range(n_bins)]

        return {
            "scores": {"audio_mean": score, "tts_like": tts_like, "hnr_proxy": hnr_proxy},
            "flags_audio": flags,
            "timeline": timeline
        }

    except Exception as e:
        return {"error": f"audio_analyze_error: {e}"}
    finally:
        # Non rimuoviamo i file temporanei in modo aggressivo per debugging;
        # se vuoi, abilita una pulizia condizionale.
        pass
