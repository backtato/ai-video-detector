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
    - Timeline binned per secondo con medie robuste.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"timeline": [], "summary": {}}

    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # campiona a min(fps, src_fps/6) per ridurre dup falsi su video molto statici
    target_fps = min(fps, max(1.0, src_fps / 6.0)) if src_fps > 0 else fps
    step = max(1, int(round(max(src_fps / target_fps, 1.0)))) if src_fps > 0 else int(round(30.0 / fps))

    frames_info = []
    prev_gray = None
    flow_mag_means = []
    prev_hash = None
    total_frames = 0

    # Lettura frame
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
        if idx % step != 0:
            continue

        ok, frame = cap.retrieve()
        if not ok or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Edge var (Laplacian)
        edge_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        y_mean   = float(np.mean(gray))

        # Motion MAD (proxy)
        motion = 0.0
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion = float(np.mean(diff))
        # Optical flow
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
            # Sgonfia dup se c'è motion reale recente
            if len(flow_mag_means) >= 3:
                recent_flow = float(np.mean(flow_mag_means[-3:]))
                if recent_flow > 0.2:
                    dup = max(0.0, dup - 0.15)
        else:
            dup = 0.0
        prev_hash = ph

        # Blockiness & banding
        blockiness = _blockiness_score(gray)
        banding    = _banding_score(gray)

        frames_info.append({
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
        return {"timeline": [], "summary": {}}

    # Binning per secondo con medie
    # Stima seconds da numero frames campionati e target_fps
    seconds = int(np.ceil(float(total_frames) / max(target_fps, 1.0)))
    seconds = max(1, min(seconds, 180))

    # segmenta frames_info in blocchi ~ per secondo
    per_sec = []
    stride = max(1, int(round(total_frames / seconds)))
    for s in range(seconds):
        start = s * stride
        end   = min(total_frames, (s + 1) * stride)
        if start >= end:
            break
        seg = frames_info[start:end]
        # medie robuste
        y_mean      = float(np.mean([x["y_mean"] for x in seg]))
        edge_var    = float(np.mean([x["edge_var"] for x in seg]))
        motion      = float(np.mean([x["motion"] for x in seg]))
        dup         = float(np.mean([x["dup"] for x in seg]))
        blockiness  = float(np.mean([x["blockiness"] for x in seg]))
        banding     = float(np.mean([x["banding"] for x in seg]))

        per_sec.append({
            "y_mean": y_mean, "edge_var": edge_var, "motion": motion,
            "dup": dup, "blockiness": blockiness, "banding": banding
        })

    # summary
    timeline = []
    def arr(key): return np.array([x[key] for x in per_sec], dtype=np.float32)

    for i, row in enumerate(per_sec):
        timeline.append({
            "start": float(i),
            "end": float(i + 1),
            **row
        })

    stats = {
        "width": int(min(src_w, src_h)) if (src_w and src_h) else 0,
        "height": int(max(src_w, src_h)) if (src_w and src_h) else 0,
        "src_fps": float(src_fps),
        "duration": float(seconds),
        "sampled_fps": float(target_fps),
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