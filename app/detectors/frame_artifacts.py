# detectors/frame_artifacts.py
import cv2
import numpy as np
from typing import Tuple, Dict

def _hf_energy(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 80, 160)
    return float(edges.mean()) / 255.0

def _blockiness(gray: np.ndarray) -> float:
    # Indice molto grezzo sul DCT-like (solo per MVP)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    g = np.abs(gx) + np.abs(gy)
    return float(np.mean(g)) / 255.0

def score_frame_artifacts(path: str, target_frames: int = 24) -> Tuple[float, Dict]:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        # fallback: campiona in loop
        n = target_frames
    step = max(1, n // target_frames)
    idxs = list(range(0, n, step))[:target_frames]

    hf_vals, blk_vals = [], []
    grabbed = 0

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))
        hf_vals.append(_hf_energy(gray))
        blk_vals.append(_blockiness(gray))
        grabbed += 1

    cap.release()

    if grabbed == 0:
        return 0.5, {"n_frames": 0, "HF": 0.0, "Blockiness": 0.0}

    hf = float(np.mean(hf_vals))
    blk = float(np.mean(blk_vals))

    # combinazione grezza
    score = 0.5 * (1 - hf) + 0.5 * blk
    score = max(0.0, min(1.0, float(score)))

    return score, {"n_frames": grabbed, "HF": round(hf, 2), "Blockiness": round(blk, 2)}
