# app/analyzers/video.py
import os, glob, json, tempfile, subprocess
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2

def _run(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)

def _sample_frames(path: str, target_fps: float = 2.5, max_frames: int = 60) -> Tuple[str, List[float]]:
    tmp = tempfile.mkdtemp(prefix="frames_")
    p = _run(["ffprobe","-v","error","-print_format","json","-show_format","-show_streams", path])
    duration = None
    try:
        info = json.loads(p.stdout or "{}")
        duration = float((info.get("format") or {}).get("duration") or 0.0)
    except: duration = None

    span = duration if (duration and duration>0) else 24.0
    fps = min(max_frames / max(span,0.1), target_fps)
    fps = max(fps, 0.5)

    _run(["ffmpeg","-hide_banner","-v","error","-i", path, "-vf", f"fps={fps:.3f}", "-qscale:v","2", os.path.join(tmp,"frame_%05d.jpg")])
    files = sorted(glob.glob(os.path.join(tmp,"frame_*.jpg")))
    times = [(i+1)*(1.0/fps) for i in range(len(files))]
    return tmp, times

def _optical_noise(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    v = float(lap.var())
    return float(np.clip(v/(v+1000.0), 0.0, 1.0))

def _edge_coherence(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1,0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0,1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    m, s = float(np.mean(mag)+1e-6), float(np.std(mag)+1e-6)
    coh = m/(s+1e-6)
    return float(np.clip(coh/(coh+5.0), 0.0, 1.0))

def _hist_cut_score(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    h1 = cv2.calcHist([prev_gray],[0],None,[64],[0,256])
    h2 = cv2.calcHist([gray],[0],None,[64],[0,256])
    h1 = cv2.normalize(h1,None).flatten()
    h2 = cv2.normalize(h2,None).flatten()
    d = float(cv2.norm(h1, h2, cv2.NORM_L2))
    return float(np.clip(d, 0.0, 1.0))

def _flicker_hz(luma: List[float], fps: float) -> Optional[float]:
    if not luma or fps<=0: return None
    x = np.array(luma, dtype=np.float32)
    x = (x - x.mean())/(x.std()+1e-6)
    X = np.fft.rfft(x); P = np.abs(X)
    freqs = np.fft.rfftfreq(len(x), d=1.0/max(fps,1e-6))
    m = (freqs>=40.0)&(freqs<=70.0)
    if not np.any(m): return None
    pf = float(freqs[m][np.argmax(P[m])])
    if abs(pf-50.0)<5.0: return 50.0
    if abs(pf-60.0)<5.0: return 60.0
    return None

def analyze_video(path: str) -> Dict[str, Any]:
    frames_dir, times = _sample_frames(path, target_fps=2.5, max_frames=60)
    files = sorted(glob.glob(os.path.join(frames_dir,"frame_*.jpg")))
    last_gray = None
    cut_count = 0
    frame_scores: List[float] = []
    luma_series: List[float] = []
    flags: List[str] = []

    try:
        for fp, t in zip(files, times):
            img = cv2.imread(fp)
            if img is None: continue
            h,w = img.shape[:2]
            scale = 256.0/max(h,w)
            if scale<1.0:
                img = cv2.resize(img,(int(w*scale),int(h*scale)), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            luma_series.append(float(np.mean(gray)))

            noise = _optical_noise(gray)
            edgec = _edge_coherence(gray)
            score = float(np.clip(0.5 + 0.25*(1.0-noise) + 0.15*(edgec-0.5), 0.0, 1.0))

            if last_gray is not None:
                if _hist_cut_score(last_gray, gray)>0.55:
                    cut_count += 1
            last_gray = gray

            frame_scores.append(score)

        fps_est = len(files)/max((times[-1] if times else 1.0), 1e-6)
        flick = _flicker_hz(luma_series, fps=fps_est)

        timeline = []
        for i,s in enumerate(frame_scores):
            t = times[i] if i<len(times) else i
            timeline.append({"start": max(t-0.5,0.0), "end": t+0.5, "ai_score": float(s)})

        scores = {"frame_mean": float(np.mean(frame_scores)) if frame_scores else 0.5,
                  "frame_std": float(np.std(frame_scores)) if frame_scores else 0.0}
        video = {"cuts": int(cut_count), "flicker_hz": flick}

        if flick in (50.0,60.0): flags.append("mains_flicker_detected")
        if (scores["frame_mean"]<0.45) and (scores["frame_std"]<0.05): flags.append("low_motion")

        return {"scores": scores, "video": video, "flags_video": flags, "timeline": timeline}
    finally:
        try:
            for f in files:
                try: os.remove(f)
                except: pass
            os.rmdir(frames_dir)
        except: pass
