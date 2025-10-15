import cv2
import numpy as np
from typing import Dict, Any, Tuple

def _high_frequency_energy(gray: np.ndarray) -> float:
    # simple Laplacian variance proxy for sharpness/edges (normalized)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var())
    # normalize by a soft factor
    return np.tanh(var / 1000.0)

def _blockiness(gray: np.ndarray) -> float:
    # very rough estimate: difference across 8x8 boundaries (JPEG-like)
    h, w = gray.shape
    v_edges = gray[:, 7::8].astype(np.float32) - gray[:, 8::8].astype(np.float32) if w > 16 else np.zeros_like(gray[:, :1], dtype=np.float32)
    h_edges = gray[7::8, :].astype(np.float32) - gray[8::8, :].astype(np.float32) if h > 16 else np.zeros_like(gray[:1, :], dtype=np.float32)
    mean_edge = float(np.mean(np.abs(v_edges))) + float(np.mean(np.abs(h_edges)))
    return np.tanh(mean_edge / 20.0)

def _noise_consistency(frames_gray: list) -> float:
    # Estimate frame-to-frame noise consistency. Synthetic videos may have atypical temporal noise.
    if len(frames_gray) < 3:
        return 0.5
    diffs = []
    for i in range(1, len(frames_gray)):
        diffs.append(float(np.mean(np.abs(frames_gray[i].astype(np.float32) - frames_gray[i-1].astype(np.float32)))))
    v = np.var(diffs) if len(diffs) > 1 else 0.0
    return 1.0 - np.tanh(v / 50.0)

def score_frame_artifacts(frames_bgr: list) -> Dict[str, Any]:
    # Returns score in [0,1] where higher leans AI-generated.
    if not frames_bgr:
        return {"score": 0.6, "notes": ["No frames extracted"]}

    hf_vals = []
    blk_vals = []
    frames_gray = []
    for f in frames_bgr:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        frames_gray.append(gray)
        hf_vals.append(_high_frequency_energy(gray))
        blk_vals.append(_blockiness(gray))

    hf = float(np.mean(hf_vals)) if hf_vals else 0.0
    blk = float(np.mean(blk_vals)) if blk_vals else 0.0
    noise = _noise_consistency(frames_gray)

    # Heuristic combination. Synthetic may show higher blockiness and atypical noise consistency with either very high or very low HF.
    raw = 0.4*blk + 0.3*(1.0 - noise) + 0.3*(1.0 - abs(hf - 0.6))  # prefer mid HF; extremes can be suspicious
    score = max(0.0, min(1.0, raw))

    notes = [
        f"HF~{hf:.2f}",
        f"Blockiness~{blk:.2f}",
        f"NoiseConsistency~{noise:.2f}"
    ]
    return {"score": score, "notes": notes}
