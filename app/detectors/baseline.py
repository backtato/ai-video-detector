import numpy as np
import cv2 as cv
from typing import List, Tuple, Dict

def variance_of_laplacian(gray):
    return float(cv.Laplacian(gray, cv.CV_64F).var())

def blockiness_score(gray, block=8):
    h, w = gray.shape
    v_edges = np.abs(np.diff(gray, axis=1))[:, ::block]
    h_edges = np.abs(np.diff(gray, axis=0))[::block, :]
    return float(np.mean(v_edges) + np.mean(h_edges)) / 255.0

def flow_temporal_incoherence(prev, curr):
    flow = cv.calcOpticalFlowFarneback(prev, curr, None,
                                       pyr_scale=0.5, levels=3, winsize=15,
                                       iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.std(mag))

class BaselineDetector:
    def __init__(self, target_fps=6, max_frames=480, min_edge_var=15.0):
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.min_edge_var = min_edge_var

    def score(self, frames: List[np.ndarray], fps: float) -> Tuple[float, Dict]:
        grays = [cv.cvtColor(f, cv.COLOR_BGR2GRAY) for f in frames]

        blurs = [variance_of_laplacian(g) for g in grays]
        blocks = [blockiness_score(g) for g in grays]

        flows = []
        for i in range(1, len(grays)):
            flows.append(flow_temporal_incoherence(grays[i-1], grays[i]))

        blur_med = float(np.median(blurs)) if blurs else 0.0
        blur_iqr = float(np.percentile(blurs, 75) - np.percentile(blurs, 25)) if blurs else 0.0
        block_mean = float(np.mean(blocks)) if blocks else 0.0
        flow_std = float(np.std(flows)) if flows else 0.0
        flow_mean = float(np.mean(flows)) if flows else 0.0

        low_texture_penalty = 1.0 if blur_med < self.min_edge_var else 0.0

        b = np.clip((20.0 / (blur_med + 1e-6)), 0, 3)
        k = np.clip(block_mean * 2.0, 0, 2.5)
        t = np.clip((flow_std + flow_mean) * 0.25, 0, 2.5)

        raw = 0.35 * b + 0.4 * k + 0.25 * t + 0.25 * low_texture_penalty
        score = float(1 / (1 + np.exp(-(raw - 1.2))))

        details = {
            "frames": len(frames),
            "fps_inferred": fps,
            "blur_median": blur_med,
            "blur_iqr": blur_iqr,
            "blockiness_mean": block_mean,
            "flow_mean": flow_mean,
            "flow_std": flow_std,
            "low_texture_penalty": low_texture_penalty,
            "raw": raw,
        }
        return score, details