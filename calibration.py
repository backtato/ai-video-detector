# calibration.py
from typing import Dict, Tuple

def combine_scores(raw: Dict[str, float], weights: Dict[str, float]) -> float:
    num = 0.0
    den = 0.0
    for k, w in weights.items():
        v = float(raw.get(k, 0.5))
        num += w * v
        den += w
    if den <= 0:
        return 0.5
    return max(0.0, min(1.0, num / den))

def calibrate(score: float, calib: Dict, duration_sec: float, n_frames: int) -> Tuple[float, float]:
    # bias/scale semplici
    s = score * float(calib.get("scale", 1.0)) + float(calib.get("bias", 0.0))
    s = max(0.0, min(1.0, s))
    # confidence cresce con frames e durata
    conf = 0.3
    if duration_sec >= 3:
        conf += 0.3
    if n_frames >= 12:
        conf += 0.4
    return s, max(0.0, min(1.0, conf))
