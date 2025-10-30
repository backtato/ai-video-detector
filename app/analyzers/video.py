import cv2, numpy as np

def _clamp(v, lo, hi): return max(lo, min(hi, v))

def _blockiness_score(gray: np.ndarray) -> float:
    v_edges = np.mean(np.abs(np.diff(gray[:, ::8].astype(np.float32), axis=1)))
    h_edges = np.mean(np.abs(np.diff(gray[::8, :].astype(np.float32), axis=0)))
    return float((v_edges + h_edges) / 510.0)

def _banding_score(gray: np.ndarray) -> float:
    q = (gray // 16) * 16
    diff = np.mean(np.abs(gray.astype(np.int16) - q.astype(np.int16)))
    return float(diff / 255.0)

def analyze(path: str, sample_seconds: int = 30, fps_cap: float = 30.0) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return {"timeline": [], "summary": {}}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = total / max(1.0, fps)
    max_frames = int(min(dur, sample_seconds) * min(fps, fps_cap))

    frames = []; idx = 0; step = max(1, int(fps // max(1.0, min(fps, fps_cap))))
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        idx += 1
        if len(frames) >= max_frames: break
    cap.release()

    n = len(frames)
    if n < 2: return {"timeline": [], "summary": {}}

    opt_mags, motions, dups, blockiness, bandings = [], [], [], [], []

    def _motion_proxy(a, b):
        return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))) / 255.0)

    for i in range(1, n):
        a, b = frames[i-1], frames[i]
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 1, 8, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        opt_mags.append(float(np.mean(mag)))
        motions.append(_motion_proxy(a, b))
        dups.append(float(np.mean((a == b).astype(np.float32))))
        blockiness.append(_blockiness_score(b))
        bandings.append(_banding_score(b))

    t = []
    for i in range(len(opt_mags)):
        s = 0.0
        s += 0.40 * _clamp(dups[i], 0.0, 1.0)
        s += 0.25 * _clamp(blockiness[i], 0.0, 1.0)
        s += 0.20 * _clamp(bandings[i], 0.0, 1.0)
        s += 0.15 * (1.0 - _clamp(motions[i], 0.0, 1.0))
        t.append(_clamp(s, 0.0, 1.0))

    summary = {
        "optflow_mag_avg": float(np.mean(opt_mags)) if opt_mags else 0.0,
        "motion_avg": float(np.mean(motions)) if motions else 0.0,
        "dup_avg": float(np.mean(dups)) if dups else 0.0,
        "blockiness_avg": float(np.mean(blockiness)) if blockiness else 0.0,
        "banding_avg": float(np.mean(bandings)) if bandings else 0.0,
    }
    return {"timeline": t, "summary": summary}
