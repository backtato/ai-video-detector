from typing import List, Tuple, Optional, Callable, Dict
import cv2
import numpy as np

# Crea finestre [t_start, t_end) di dimensione window_sec, senza sovrapposizione
def make_segments(total_duration: float, window_sec: int = 3) -> List[Tuple[float, float]]:
    if total_duration <= 0 or window_sec <= 0:
        return []
    segs = []
    t = 0.0
    while t + window_sec <= total_duration + 1e-6:
        segs.append((t, t + window_sec))
        t += window_sec
    if not segs and total_duration > 0:
        segs.append((0.0, min(total_duration, float(window_sec))))
    return segs

# Analisi rapida su finestre: estrai pochi frame equidistanti per finestra,
# applica prefilter opzionale e calcola 2 semplici indici (blur + blockiness)
# -> questi NON sostituiscono i tuoi detectors; servono come diagnostica nel JSON.
def analyze_segments(path: str,
                     segments: List[Tuple[float, float]],
                     prefilter_fn: Optional[Callable] = None) -> List[Dict]:
    if not segments:
        return []

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    dur = (total / fps) if (fps > 0 and total > 0) else 0.0

    results = []
    for (ts, te) in segments:
        # 3 campioni per finestra
        at = []
        for frac in (0.2, 0.5, 0.8):
            t_pick = ts + (te - ts) * frac
            if dur > 0 and fps > 0:
                frame_idx = int(t_pick * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if prefilter_fn:
                try:
                    frame = prefilter_fn(frame)
                except Exception:
                    pass
            at.append(_quick_metrics(frame))

        if at:
            # media dei campioni
            blur = float(np.mean([x["blur"] for x in at]))
            block = float(np.mean([x["blockiness"] for x in at]))
            # euristica per un mini-score “frame_artifacts_like”
            # (solo diagnostico: non influisce sull'ai_score globale)
            est = max(0.0, min(1.0, 0.5 + 0.15 * (block - 0.5) - 0.1 * (blur - 0.5)))
        else:
            blur, block, est = 0.5, 0.5, 0.5

        results.append({
            "t_start": round(ts, 3),
            "t_end": round(te, 3),
            "quick_blur": round(blur, 3),
            "quick_blockiness": round(block, 3),
            "quick_artifacts_estimate": round(est, 3),
        })

    cap.release()
    return results

def _quick_metrics(frame_bgr) -> Dict[str, float]:
    # blur: Varianza del Laplaciano (normalizzata)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    var_lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur = _norm(var_lap, lo=50, hi=600)

    # blockiness: differenze medie sui confini 8x8 (proxy JPEG/MPEG)
    h, w = gray.shape
    # bordi verticali ogni 8 px
    diffs = []
    for x in range(8, w, 8):
        col1 = gray[:, x-1].astype(np.float32)
        col2 = gray[:, x].astype(np.float32)
        diffs.append(float(np.mean(np.abs(col2 - col1))))
    blockiness = _norm(np.mean(diffs) if diffs else 0.0, lo=2.0, hi=12.0)

    return {"blur": blur, "blockiness": blockiness}

def _norm(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    v = (x - lo) / (hi - lo)
    return float(max(0.0, min(1.0, v)))
