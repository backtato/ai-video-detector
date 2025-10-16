import cv2
import numpy as np
from typing import Dict, Any

def _high_freq_energy(gray: np.ndarray) -> float:
    # Use Laplacian variance as HF proxy
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _blockiness(gray: np.ndarray) -> float:
    # Simple blockiness: mean absolute diff across 8x8 boundaries
    h, w = gray.shape[:2]
    v_edges = np.mean(np.abs(np.diff(gray[:, ::8], axis=1)))
    h_edges = np.mean(np.abs(np.diff(gray[::8, :], axis=0)))
    return float((v_edges + h_edges) / 2.0)

def _noise_consistency(gray: np.ndarray) -> float:
    # Estimate local noise std across patches and see variance of std → higher consistency → more "AI-like" sometimes
    h, w = gray.shape[:2]
    patch = 32
    sigmas = []
    for y in range(0, h - patch, patch):
        for x in range(0, w - patch, patch):
            region = gray[y:y+patch, x:x+patch]
            sigmas.append(float(region.std()))
    if not sigmas:
        return 0.0
    return float(np.std(sigmas))

def score_frame_artifacts(path: str, target_frames: int = 32) -> Dict[str, Any]:
    """
    Sample up to target_frames uniformly, compute simple artifact metrics.
    Combine into [0..1] AI-likelihood proxy (heuristic).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"score": 0.5, "details": {"error": "cannot_open"}}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration = float(cap.get(cv2.CAP_PROP_DURATION)) if hasattr(cv2, 'CAP_PROP_DURATION') else 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)

    if total <= 0 and fps > 0 and duration > 0:
        total = int(fps * duration)

    idxs = np.linspace(0, max(0, total-1), num=min(target_frames, max(1, total)), dtype=np.int32)
    hf_vals, blk_vals, noise_vals = [], [], []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hf_vals.append(_high_freq_energy(gray))
        blk_vals.append(_blockiness(gray))
        noise_vals.append(_noise_consistency(gray))

    cap.release()

    if not hf_vals:
        return {"score": 0.5, "details": {"error": "no_frames"}}

    # Normalize metrics
    def nzmean(x): return float(np.mean(x)) if len(x) else 0.0
    hf, blk, noi = nzmean(hf_vals), nzmean(blk_vals), nzmean(noise_vals)

    # Scale to [0..1] with soft thresholds
    def scale(x, lo, hi):
        if x <= lo: return 0.0
        if x >= hi: return 1.0
        return float((x - lo) / (hi - lo))

    # Empirical ranges (tunable)
    hf_s = scale(hf, 50, 400)      # very smooth → higher AI suspicion
    blk_s = scale(blk, 2.0, 10.0)  # block boundaries → compression/generation
    noi_s = scale(noi, 0.5, 5.0)   # uniform noise across patches may indicate synthesis

    # Combine: more smoothness (low HF) → AI, so invert hf_s
    hf_inv = 1.0 - hf_s
    score = 0.5 * hf_inv + 0.3 * blk_s + 0.2 * noi_s
    return {
        "score": max(0.0, min(1.0, float(score))),
        "details": {
            "HF_mean": hf,
            "Blockiness_mean": blk,
            "NoiseConsistency_std": noi,
            "frames_used": int(len(hf_vals)),
            "width": w, "height": h, "fps": fps, "duration": duration, "frame_count": total,
        }
    }
