import os
import subprocess
import tempfile
import numpy as np
import soundfile as sf

def _extract_wav_16k(path: str):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    cmd = ["ffmpeg","-y","-i",path,"-ac","1","-ar","16000","-f","wav",tmp.name]
    # non sollevare eccezione: catturiamo noi gli errori
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg_convert_failed")
    try:
        wav, sr = sf.read(tmp.name, dtype="float32", always_2d=False)
    except Exception as e:
        # pulizia e segnala
        try: os.unlink(tmp.name)
        except Exception: pass
        raise RuntimeError("soundfile_read_failed")
    return tmp.name, wav, sr

def analyze(path: str, meta: dict):
    # estrazione 16k mono
    tmp = None
    try:
        tmp, wav, sr = _extract_wav_16k(path)
        if wav.ndim>1:
            wav = wav[:,0]
        dur = len(wav)/sr if sr>0 else 0.0

        # finestre 0.5s
        win = int(sr*0.5) if sr else 1
        win = max(1, win)

        rms = []
        zcr = []
        flat = []
        roll = []
        for i in range(0, len(wav), win):
            seg = wav[i:i+win]
            if len(seg)==0: continue
            rms.append(float(np.sqrt((seg**2).mean())))
            zc = np.mean(np.abs(np.diff(np.sign(seg))))/2.0
            zcr.append(float(zc))
            # spettrali
            spec = np.fft.rfft(seg * np.hanning(len(seg)))
            mag = np.abs(spec) + 1e-9
            flat.append(float(np.exp(np.mean(np.log(mag)))/np.mean(mag)))
            cutoff = 0.85*np.sum(mag)
            s = 0.0
            idx = 0
            for k, m in enumerate(mag):
                s += m
                if s>=cutoff:
                    idx = k; break
            roll.append(float(idx)/max(1.0, len(mag)))

        rms_arr = np.array(rms) if rms else np.zeros(1)
        zcr_arr = np.array(zcr) if zcr else np.zeros(1)
        flat_arr = np.array(flat) if flat else np.zeros(1)
        roll_arr = np.array(roll) if roll else np.zeros(1)

        speech_thr = np.percentile(rms_arr, 60) if len(rms_arr)>0 else 0.0
        speech_ratio = float(np.mean(rms_arr >= speech_thr)) if len(rms_arr)>0 else 0.0

        tts_like = float(min(1.0,
            0.6*float(np.mean(flat_arr)) +
            0.2*(1.0/(1e-6+float(np.var(zcr_arr)))) +
            0.2*(1.0/(1e-6+float(np.var(roll_arr)))))
        )

        def norm01(x):
            x=np.clip(x,0,None)
            if len(x)==0: return np.zeros(1)
            mx = np.max(x); mn = np.min(x)
            return (x-mn)/ (mx-mn+1e-9)

        # timeline ai-suspicion audio
        dzcr = np.diff(np.concatenate([[zcr_arr[0] if len(zcr_arr)>0 else 0.0], zcr_arr])) if len(zcr_arr)>0 else np.zeros(1)
        droll = np.diff(np.concatenate([[roll_arr[0] if len(roll_arr)>0 else 0.0], roll_arr])) if len(roll_arr)>0 else np.zeros(1)
        tline = 0.5*norm01(flat_arr) + 0.3*(1.0-norm01(dzcr**2)) + 0.2*(1.0-norm01(np.abs(droll)))
        tline = np.clip(tline, 0.0, 1.0).tolist()

        flags = {
            "speech_ratio": speech_ratio,
            "tts_like": tts_like,
            "rms_var": float(np.var(rms_arr)) if len(rms_arr)>0 else 0.0,
            "zcr_var": float(np.var(zcr_arr)) if len(zcr_arr)>0 else 0.0,
            "roll_var": float(np.var(roll_arr)) if len(roll_arr)>0 else 0.0,
        }

        # clamp alla durata (in secondi)
        tlen = int(max(1, round(dur)))
        tline = tline[:tlen] if len(tline)>=tlen else (tline + [tline[-1] if tline else 0.5]*(tlen-len(tline)))

        return {
            "scores": {
                "speech_ratio": speech_ratio,
                "tts_like": tts_like,
            },
            "flags_audio": flags,
            "timeline": tline
        }
    except Exception as e:
        # fallback neutrale (non solleva: evita 500)
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