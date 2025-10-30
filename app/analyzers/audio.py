import subprocess
import tempfile
import numpy as np
import soundfile as sf

def _extract_wav_16k(path: str):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    cmd = ["ffmpeg","-y","-i",path,"-ac","1","-ar","16000","-f","wav",tmp.name]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    wav, sr = sf.read(tmp.name, dtype="float32", always_2d=False)
    return tmp.name, wav, sr

def _safe_var(x):
    if len(x)==0: return 0.0
    return float(np.var(x))

def analyze(path: str, meta: dict):
    # extract 16k mono
    tmp, wav, sr = _extract_wav_16k(path)
    try:
        if wav.ndim>1:
            wav = wav[:,0]
        dur = len(wav)/sr if sr>0 else 0.0
        # features windows of 0.5s
        win = int(sr*0.5)
        if win<=0: win=1
        rms = []
        zcr = []
        flat = []
        roll = []
        # simple speech activity
        for i in range(0, len(wav), win):
            seg = wav[i:i+win]
            if len(seg)==0: continue
            rms.append(float(np.sqrt((seg**2).mean())))
            zc = np.mean(np.abs(np.diff(np.sign(seg))))/2.0
            zcr.append(float(zc))
            # spectral features
            spec = np.fft.rfft(seg * np.hanning(len(seg)))
            mag = np.abs(spec) + 1e-9
            flat.append(float(np.exp(np.mean(np.log(mag)))/np.mean(mag)))
            # rolloff ~ 85%
            cutoff = 0.85*np.sum(mag)
            s = 0.0
            idx = 0
            for k, m in enumerate(mag):
                s += m
                if s>=cutoff:
                    idx = k; break
            roll.append(float(idx)/max(1.0, len(mag)))
        # ratios
        rms_arr = np.array(rms) if rms else np.zeros(1)
        zcr_arr = np.array(zcr) if zcr else np.zeros(1)
        flat_arr = np.array(flat) if flat else np.zeros(1)
        roll_arr = np.array(roll) if roll else np.zeros(1)

        speech_thr = np.percentile(rms_arr, 60) if rms else 0.0
        speech_ratio = float(np.mean(rms_arr >= speech_thr)) if rms else 0.0

        # TTS-likeness proxy: high flatness + low zcr variance + stable rolloff
        tts_like = float(min(1.0, 0.6*float(np.mean(flat_arr)) + 0.2*(1.0/(1e-6+float(np.var(zcr_arr)))) + 0.2*(1.0/(1e-6+float(np.var(roll_arr))))))

        # timeline score ~ suspicion of AI in audio
        # normalize each feature to [0,1]
        def norm01(x): 
            x=np.clip(x,0,None); 
            if len(x)==0: return np.zeros(1)
            mx = np.max(x); mn = np.min(x)
            return (x-mn)/ (mx-mn+1e-9)
        tline = 0.5*norm01(flat_arr) + 0.3*(1.0-norm01(np.diff(np.concatenate([[zcr_arr[0]], zcr_arr]))**2)) + 0.2*(1.0-norm01(np.abs(np.diff(np.concatenate([[roll_arr[0]], roll_arr])))))
        tline = np.clip(tline, 0.0, 1.0).tolist()

        flags = {
            "speech_ratio": speech_ratio,
            "tts_like": tts_like,
            "rms_var": float(np.var(rms_arr)) if rms else 0.0,
            "zcr_var": float(np.var(zcr_arr)) if zcr else 0.0,
            "roll_var": float(np.var(roll_arr)) if roll else 0.0,
        }
        return {
            "scores": {
                "speech_ratio": speech_ratio,
                "tts_like": tts_like,
            },
            "flags_audio": flags,
            "timeline": tline[: int(max(1, round(dur)))]
        }
    finally:
        try: import os; os.unlink(tmp)
        except Exception: pass