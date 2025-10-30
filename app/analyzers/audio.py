import os
import subprocess
import tempfile
import numpy as np
import soundfile as sf

def _extract_wav_16k(path: str):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    cmd = ["ffmpeg","-y","-i",path,"-ac","1","-ar","16000","-f","wav",tmp.name]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg_convert_failed")
    try:
        wav, sr = sf.read(tmp.name, dtype="float32", always_2d=False)
    except Exception:
        try: os.unlink(tmp.name)
        except Exception: pass
        raise RuntimeError("soundfile_read_failed")
    return tmp.name, wav, sr

def _norm01(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.zeros(1)
    mn, mx = float(np.min(x)), float(np.max(x))
    return (x - mn) / (mx - mn + 1e-9)

def analyze(path: str, meta: dict):
    tmp = None
    try:
        tmp, wav, sr = _extract_wav_16k(path)
        if wav.ndim > 1:
            wav = wav[:, 0]
        dur = len(wav)/sr if sr > 0 else 0.0

        win = max(1, int(sr * 0.5)) if sr else 1
        rms, zcr, flat, roll, sc_cent = [], [], [], [], []

        for i in range(0, len(wav), win):
            seg = wav[i:i+win]
            if len(seg) == 0:
                continue
            rms.append(float(np.sqrt((seg**2).mean())))
            zc = np.mean(np.abs(np.diff(np.sign(seg))))/2.0
            zcr.append(float(zc))
            winseg = seg * np.hanning(len(seg))
            spec = np.fft.rfft(winseg)
            mag = np.abs(spec) + 1e-9
            flat.append(float(np.exp(np.mean(np.log(mag))) / np.mean(mag)))
            cutoff = 0.85 * np.sum(mag)
            s = 0.0
            idx = 0
            for k, m in enumerate(mag):
                s += m
                if s >= cutoff:
                    idx = k; break
            roll.append(float(idx) / max(1.0, len(mag)))
            freqs = np.linspace(0.0, 1.0, len(mag))
            sc = float(np.sum(freqs * mag) / np.sum(mag))
            sc_cent.append(sc)

        rms_arr  = np.array(rms)  if rms  else np.zeros(1)
        zcr_arr  = np.array(zcr)  if zcr  else np.zeros(1)
        flat_arr = np.array(flat) if flat else np.zeros(1)
        roll_arr = np.array(roll) if roll else np.zeros(1)
        sc_arr   = np.array(sc_cent) if sc_cent else np.zeros(1)

        speech_thr = np.percentile(rms_arr, 60) if rms_arr.size else 0.0
        speech_ratio = float(np.mean(rms_arr >= speech_thr)) if rms_arr.size else 0.0

        flat_mean = float(np.mean(flat_arr)) if flat_arr.size else 0.0
        sc_var    = float(np.var(sc_arr))    if sc_arr.size  else 0.0
        roll_var  = float(np.var(roll_arr))  if roll_arr.size else 0.0
        zcr_var   = float(np.var(zcr_arr))   if zcr_arr.size else 0.0

        tts_base = 0.7 * flat_mean + 0.15 * (1.0/(1e-6 + zcr_var)) + 0.15 * (1.0/(1e-6 + roll_var))
        attenuation = 1.0 / (1.0 + 5.0 * (sc_var + roll_var + zcr_var))
        tts_like = float(np.clip(tts_base * attenuation, 0.0, 1.0))

        # Cap del TTS se la variabilità non è trascurabile
        variability = sc_var + roll_var + zcr_var
        if variability > 0.005:
            tts_like = float(min(tts_like, 0.90))

        dzcr  = np.diff(np.concatenate([[zcr_arr[0] if zcr_arr.size else 0.0], zcr_arr])) if zcr_arr.size else np.zeros(1)
        droll = np.diff(np.concatenate([[roll_arr[0] if roll_arr.size else 0.0], roll_arr])) if roll_arr.size else np.zeros(1)
        tline = 0.5*_norm01(flat_arr) + 0.3*(1.0-_norm01(dzcr**2)) + 0.2*(1.0-_norm01(np.abs(droll)))
        tline = np.clip(tline, 0.0, 1.0).tolist()

        tlen = int(max(1, round(dur)))
        if len(tline) < tlen:
            tline = tline + [tline[-1] if tline else 0.5] * (tlen - len(tline))
        else:
            tline = tline[:tlen]

        return {
            "scores": {
                "speech_ratio": speech_ratio,
                "tts_like": tts_like,
            },
            "flags_audio": {
                "speech_ratio": speech_ratio,
                "tts_like": tts_like,
                "rms_var": float(np.var(rms_arr)) if rms_arr.size else 0.0,
                "zcr_var": zcr_var,
                "roll_var": roll_var,
                "sc_var": sc_var,
            },
            "timeline": tline
        }
    except Exception as e:
        tlen = int(max(1, round(meta.get("duration") or 0.0)))
        return {
            "scores": {},
            "flags_audio": {"error": str(e)},
            "timeline": [0.5]*tlen
        }
    finally:
        if tmp:
            try: os.unlink(tmp)
            except Exception: pass