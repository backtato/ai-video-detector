# app/analyzers/video.py
# Estensione: optical flow, blockiness, banding, duplicate-frames, sampling adattivo.
# Non rimuove funzionalità preesistenti: mantiene summary, timeline e campi classici.

import cv2
import numpy as np

def _blockiness_score(gray: np.ndarray) -> float:
    """Proxy semplice per blockiness (macroblocchi 8x8)."""
    h, w = gray.shape[:2]
    v_edges = np.mean(np.abs(np.diff(gray[:, ::8].astype(np.float32), axis=1)))
    h_edges = np.mean(np.abs(np.diff(gray[::8, :].astype(np.float32), axis=0)))
    return float((v_edges + h_edges) / 510.0)  # normalizzazione grezza

def _banding_score(gray: np.ndarray) -> float:
    """Proxy per banding: quantizzazione a 16 livelli e differenza medio-assoluta."""
    q = (gray // 16) * 16
    return float(np.mean(np.abs(gray.astype(np.float32) - q.astype(np.float32))) / 16.0)

def analyze(path: str, max_seconds: float = 30.0, fps: float = 2.5) -> dict:
    """
    Analisi video leggera:
    - Sampling adattivo in base a src_fps.
    - Feature per frame: y_mean, edge_var (Laplacian var), motion MAD, dup-hash,
      blockiness, banding.
    - Optical flow Farnebäck (magnitudo media tra frame consecutivi).
    - Timeline binned per secondo + summary.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"error": "opencv_open_failed"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) / src_fps) if src_fps > 0 else 0.0

    # Sampling adattivo
    step = max(int(round(src_fps / fps)) if src_fps > 0 else int(round(30 / fps)), 1)

    frames_info = []
    total_frames = 0
    grabbed = -1
    prev_gray = None
    prev_hash = None
    flow_mag_means = []

    while True:
        grabbed += 1
        ok = cap.grab()
        if not ok:
            break
        if grabbed % step != 0:
            continue
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        y_mean = float(gray.mean())
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        edge_var = float(lap.var())

        # Motion MAD
        motion = 0.0
        if prev_gray is not None:
            motion = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))

        # Optical flow Farnebäck
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_mag_means.append(float(np.mean(mag)))
        prev_gray = gray

        # Perceptual dup-hash semplice (32x32 + threshold binary)
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        ph = (small > small.mean()).astype(np.uint8)
        if prev_hash is not None:
            # similarità: 1.0 = identico
            dup = 1.0 - float(np.mean(np.bitwise_xor(ph, prev_hash)))
        else:
            dup = 0.0
        prev_hash = ph

        # Compression artifacts
        blockiness = _blockiness_score(gray)
        banding = _banding_score(gray)

        # Timestamp
        ts = (grabbed / (src_fps if src_fps > 0 else fps)) if src_fps > 0 else (len(frames_info) / fps)

        frames_info.append({
            "t": ts,
            "y_mean": y_mean,
            "edge_var": edge_var,
            "motion": motion,
            "dup": dup,
            "blockiness": blockiness,
            "banding": banding
        })
        total_frames += 1
        # Fino a circa max_seconds + margine
        if total_frames >= int(fps * max_seconds) + 90:
            break

    cap.release()

    if not frames_info:
        return {"error": "no_frames"}

    # Timeline per secondo
    last_t = frames_info[-1]["t"]
    n_bins = max(int(last_t) + 1, 1)
    timeline = []
    for sec in range(n_bins):
        seg = [x for x in frames_info if sec <= x["t"] < sec + 1]
        if not seg:
            continue
        y_mean = float(np.mean([x["y_mean"] for x in seg]))
        edge_var = float(np.mean([x["edge_var"] for x in seg]))
        motion = float(np.mean([x["motion"] for x in seg]))
        dup = float(np.mean([x["dup"] for x in seg]))
        blockiness = float(np.mean([x["blockiness"] for x in seg]))
        banding = float(np.mean([x["banding"] for x in seg]))
        timeline.append({
            "start": float(sec), "end": float(sec + 1),
            "y_mean": y_mean, "edge_var": edge_var, "motion": motion,
            "dup": dup, "blockiness": blockiness, "banding": banding
        })

    def arr(key):
        return np.array([x[key] for x in timeline], dtype=np.float32) if timeline else np.array([], np.float32)

    stats = {
        "width": width, "height": height,
        "src_fps": src_fps, "duration": duration,
        "sampled_fps": fps,
        "frames_sampled": int(total_frames),
        "timeline": timeline,
        "summary": {
            "y_mean_avg": float(arr("y_mean").mean()) if timeline else 0.0,
            "edge_var_avg": float(arr("edge_var").mean()) if timeline else 0.0,
            "motion_avg": float(arr("motion").mean()) if timeline else 0.0,
            "dup_avg": float(arr("dup").mean()) if timeline else 0.0,
            "blockiness_avg": float(arr("blockiness").mean()) if timeline else 0.0,
            "banding_avg": float(arr("banding").mean()) if timeline else 0.0,
            "optflow_mag_avg": float(np.mean(flow_mag_means)) if len(flow_mag_means) else 0.0
        }
    }
    return stats

