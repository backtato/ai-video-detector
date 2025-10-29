import cv2
import numpy as np

def _clamp(v, lo, hi): 
    return max(lo, min(hi, v))

def _blockiness_score(gray: np.ndarray) -> float:
    h, w = gray.shape[:2]
    v_edges = np.mean(np.abs(np.diff(gray[:, ::8].astype(np.float32), axis=1)))
    h_edges = np.mean(np.abs(np.diff(gray[::8, :].astype(np.float32), axis=0)))
    return float((v_edges + h_edges) / 510.0)

def _banding_score(gray: np.ndarray) -> float:
    q = (gray // 16) * 16
    diff = np.mean(np.abs(gray.astype(np.int16) - q.astype(np.int16)))
    return float(diff / 255.0)

def _optflow_mag(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))

def analyze(cv_path: str, src_fps: float, duration: float, max_seconds: int = 16) -> dict:
    """
    Campionamento adattivo (fps campionati più alti su 30fps) per ridurre falsi duplicati.
    Dup attenuato quando c'è motion/flow presente.
    """
    cap = cv2.VideoCapture(cv_path)
    if not cap.isOpened():
        return {"timeline": [], "summary": {}}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or src_fps or 0.0)
    fps = fps if fps > 0 else (src_fps or 30.0)

    # sampled_fps adattivo: più alto per 30fps (riduce dup artefatti)
    sampled_fps = 5.0 if fps >= 30.0 else 2.5
    step = max(1, int(round(fps / sampled_fps)))

    frames = []
    idx = 0
    read = 0
    limit_frames = int(min(duration, max_seconds) * fps)

    while True:
        ret = cap.grab()
        if not ret:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            read += 1
        idx += 1
        if limit_frames and idx >= limit_frames:
            break
    cap.release()

    n = len(frames)
    if n < 2:
        return {"timeline": [], "summary": {}}

    timeline = []
    opt_mags = []
    motions = []
    dups = []
    blockiness = []
    bandings = []

    def _motion_proxy(a, b):
        # differenza frame-frame
        return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)))) / 255.0 * 100.0

    for i in range(1, n):
        prev = frames[i - 1]
        cur = frames[i]
        mag = _optflow_mag(prev, cur)
        opt_mags.append(mag)

        mot = _motion_proxy(prev, cur)
        motions.append(mot)

        # duplicato se quasi identico (soglia adattiva: più permissiva se motion/flow presenti)
        diff = np.mean(np.abs(prev.astype(np.int16) - cur.astype(np.int16))) / 255.0
        dup = 1.0 if diff < 0.01 else 0.0
        dups.append(dup)

        blockiness.append(_blockiness_score(cur))
        bandings.append(_banding_score(cur))

    flow_avg = float(np.mean(opt_mags)) if opt_mags else 0.0
    motion_avg = float(np.mean(motions)) if motions else 0.0

    # Attenua dup quando c'è movimento/flow (riduce falsi duplicati su scene stabili)
    if flow_avg > 1.0 or motion_avg > 22.0:
        dups = [d * 0.6 for d in dups]  # attenuazione 40%

    dup_avg = float(np.mean(dups)) if dups else 0.0
    block_avg = float(np.mean(blockiness)) if blockiness else 0.0
    band_avg = float(np.mean(bandings)) if bandings else 0.0

    # score video per-secondo (semplice ma stabile)
    # dup ha peso ridotto; motion basso + banding/blockiness alti spingono su AI
    v_scores = []
    # normalizza alcune scale
    m_norm = _clamp(motion_avg / 40.0, 0.0, 1.0)
    b_norm = _clamp(block_avg / 0.5, 0.0, 1.0)
    ba_norm = _clamp(band_avg / 0.6, 0.0, 1.0)
    d_norm = _clamp(dup_avg, 0.0, 1.0)

    v_base = 0.45 + 0.10 * (0.5 - m_norm) + 0.12 * b_norm + 0.08 * ba_norm + 0.15 * d_norm
    v_base = _clamp(v_base, 0.25, 0.75)

    # costruisci timeline ai (per secondo)
    secs = int(np.ceil(min(duration, len(dups) / sampled_fps)))
    for s in range(secs):
        v_scores.append(v_base)
        timeline.append({"start": s, "end": s + 1, "ai_score": float(v_base)})

    summary = {
        "y_mean_avg": None,
        "edge_var_avg": None,
        "motion_avg": float(motion_avg),
        "dup_avg": float(dup_avg),
        "blockiness_avg": float(block_avg),
        "banding_avg": float(band_avg),
        "optflow_mag_avg": float(flow_avg),
        "sampled_fps": float(sampled_fps),
    }
    return {
        "timeline": timeline,
        "summary": summary
    }
