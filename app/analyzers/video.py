# app/analyzers/video.py
import os
import glob
import json
import tempfile
import subprocess
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np

def _run(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)

def _sample_frames(path: str, target_fps: float = 2.5, max_frames: int = 60) -> Tuple[str, List[float]]:
    """
    Estrae frame JPEG temporanei usando ffmpeg con fps ridotto (default ~2-3 fps).
    Ritorna cartella frames + lista tempi (secondi).
    """
    tmp = tempfile.mkdtemp(prefix="frames_")
    # Prima leggi durata (per avere tempi coerenti)
    meta = _run([
        "ffprobe","-v","error","-print_format","json",
        "-show_format","-show_streams", path
    ])
    duration = None
    try:
        info = json.loads(meta.stdout or "{}")
        duration = float((info.get("format") or {}).get("duration") or 0.0)
    except Exception:
        duration = None

    # Calcolo fps effettivo per non superare max_frames
    span = duration if (duration and duration > 0) else 24.0
    fps = min(max_frames / max(span, 0.1), target_fps)
    fps = max(fps, 0.5)

    cmd = [
        "ffmpeg","-hide_banner","-v","error",
        "-i", path,
        "-vf", f"fps={fps:.3f}",
        "-qscale:v","2",
        os.path.join(tmp, "frame_%05d.jpg")
    ]
    _run(cmd)
    files = sorted(glob.glob(os.path.join(tmp, "frame_*.jpg")))
    times = [(i+1) * (1.0/fps) for i in range(len(files))]
    return tmp, times

def _optical_noise(gray: np.ndarray) -> float:
    """
    Rumore “ottico” grezzo: varianza del Laplaciano normalizzata.
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    v = float(lap.var())
    return float(np.clip(v / (v + 1000.0), 0.0, 1.0))

def _edge_coherence(gray: np.ndarray) -> float:
    """
    Coerenza dei bordi: rapporto tra magnitudine media del gradiente e la sua varianza.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    m = float(np.mean(mag) + 1e-6)
    s = float(np.std(mag) + 1e-6)
    # più coerenza → più “naturale”
    coh = m / (s + 1e-6)
    return float(np.clip(coh / (coh + 5.0), 0.0, 1.0))

def _hist_cut_score(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """
    Cut detection (semplice): differenza di istogrammi normalizzata.
    """
    h1 = cv2.calcHist([prev_gray],[0],None,[64],[0,256])
    h2 = cv2.calcHist([gray],[0],None,[64],[0,256])
    h1 = cv2.normalize(h1, None).flatten()
    h2 = cv2.normalize(h2, None).flatten()
    d = float(cv2.norm(h1, h2, cv2.NORM_L2))
    return float(np.clip(d, 0.0, 1.0))

def _flicker_hz(luma_series: List[float], fps: float) -> Optional[float]:
    """
    Stima estremamente grezza di flicker tramite spettri sulla luminanza media.
    Nota: con fps basso la mappatura 50/60 Hz è indicativa → ritorna 50/60 se picchi normalizzati > soglie.
    """
    if not luma_series or fps <= 0:
        return None
    x = np.array(luma_series, dtype=np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    X = np.fft.rfft(x)
    P = np.abs(X)
    freqs = np.fft.rfftfreq(len(x), d=1.0/max(fps, 1e-6))
    # Cerca massimi in banda [40..70] Hz, ma molto indicativo a fps bassi
    mask = (freqs >= 40.0) & (freqs <= 70.0)
    if not np.any(mask):
        return None
    sub_f = freqs[mask]
    sub_p = P[mask]
    if len(sub_p) == 0:
        return None
    peak_i = int(np.argmax(sub_p))
    pf = float(sub_f[peak_i])
    # Snappa a 50 o 60 se vicino
    if abs(pf - 50.0) < 5.0:
        return 50.0
    if abs(pf - 60.0) < 5.0:
        return 60.0
    return None

def analyze_video(path: str) -> Dict[str, Any]:
    """
    Estrae frame, calcola euristiche frame-level e indicatori (cut, flicker, ecc.).
    Ritorna:
      - scores: frame_mean, frame_std
      - video: cuts, flicker_hz
      - flags_video: elenco di hint
      - timeline: [{start,end,ai_score}], timeline_binned calcolata altrove
    """
    frames_dir, times = _sample_frames(path, target_fps=2.5, max_frames=60)
    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    last_gray = None
    cut_count = 0
    frame_scores: List[float] = []
    luma_series: List[float] = []
    flags: List[str] = []

    try:
        for fp, t in zip(files, times):
            img = cv2.imread(fp)
            if img is None:
                continue
            h, w = img.shape[:2]
            # ridimensiona per robustezza
            scale = 256.0 / max(h, w)
            if scale < 1.0:
                img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Luma media per flicker (indicativo)
            luma_series.append(float(np.mean(gray)))

            # Heuristics
            noise = _optical_noise(gray)          # [0..1] alto=rumoroso
            edgec = _edge_coherence(gray)         # [0..1] alto=coerente
            # euristica: immagine “troppo pulita” e bordi “troppo coerenti” → leggero bias AI
            score = float(np.clip(0.5 + 0.25*(1.0 - noise) + 0.15*(edgec - 0.5), 0.0, 1.0))

            if last_gray is not None:
                cut = _hist_cut_score(last_gray, gray)
                if cut > 0.55:
                    cut_count += 1
            last_gray = gray
            frame_scores.append(score)

        fps_est = len(files) / max((times[-1] if times else 1.0), 1e-6)
        flick = _flicker_hz(luma_series, fps=fps_est)

        timeline = []
        for i, s in enumerate(frame_scores):
            t = times[i] if i < len(times) else i
            timeline.append({"start": max(t-0.5, 0.0), "end": t+0.5, "ai_score": float(s)})

        scores = {
            "frame_mean": float(np.mean(frame_scores)) if frame_scores else 0.5,
            "frame_std": float(np.std(frame_scores)) if frame_scores else 0.0,
        }
        video = {
            "cuts": int(cut_count),
            "flicker_hz": flick,
        }

        # Flags indicative
        if flick in (50.0, 60.0):
            flags.append("mains_flicker_detected")
        if (scores["frame_mean"] < 0.45) and (scores["frame_std"] < 0.05):
            flags.append("low_motion")

        return {
            "scores": scores,
            "video": video,
            "flags_video": flags,
            "timeline": timeline
        }
    finally:
        try:
            # pulizia frames
            for f in files:
                try: os.remove(f)
                except Exception: pass
            os.rmdir(frames_dir)
        except Exception:
            pass