import os, glob
import cv2
import numpy as np
from typing import Tuple, List

def _frame_score(img: np.ndarray) -> float:
    if img is None or img.size == 0:
        return 0.5
    h, w = img.shape[:2]
    scale = 256.0 / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharp = float(lap.var())
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift) + 1e-6
    logmag = np.log(mag)
    spec_mean = float(logmag.mean())
    sharp_n = np.tanh(sharp / 200.0)
    spec_n  = np.clip((spec_mean - 5.0)/3.0, 0.0, 1.0)
    score = 0.6*(1.0 - sharp_n) + 0.4*(1.0 - spec_n)
    return float(np.clip(score, 0.0, 1.0))

def video_frame_scores(frames_dir: str, times: List[float]) -> Tuple[List[float], List[dict]]:
    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    scores, timeline = [], []
    for i, fp in enumerate(files):
        img = cv2.imread(fp)
        s = _frame_score(img)
        scores.append(s)
        t = times[i] if i < len(times) else i
        timeline.append({"start": float(max(t-0.5, 0.0)), "end": float(t+0.5), "ai_score": float(s)})
    if not scores:
        scores = [0.5]
        timeline = [{"start":0.0,"end":1.0,"ai_score":0.5}]
    return scores, timeline