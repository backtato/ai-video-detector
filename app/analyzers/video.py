import cv2
import numpy as np

def analyze(path: str, max_seconds: float = 30.0, fps: float = 2.5) -> dict:
    """
    Estrae statistiche video in modo leggero e conservativo:
    - Campiona fino a ~30s a 2.5fps (≈ 75 frame max).
    - Calcola:
      * luminanza media per frame
      * varianza del Laplaciano (edge/texture)
      * differenza assoluta media tra frame consecutivi (motion proxy)
    - Ritorna timeline per secondo (media dei frame caduti in quel secondo).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"error": "opencv_open_failed"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) / src_fps) if src_fps > 0 else 0.0

    # Limita l'intervallo
    max_dur = min(max_seconds, duration if duration > 0 else max_seconds)
    sample_step = max(1, int(round((src_fps / max(0.1, fps))))) if src_fps > 0 else 1

    frames_info = []
    prev_gray = None
    total_frames = 0
    grabbed = 0

    # Leggiamo fino a max_dur seconds
    max_frames_to_read = int(src_fps * max_dur) if src_fps > 0 else int(fps * max_dur * 2)

    while grabbed < max_frames_to_read:
        ok, frame = cap.read()
        if not ok:
            break
        grabbed += 1

        # Sottocampionamento
        if src_fps > 0 and (grabbed % sample_step != 0):
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        y_mean = float(gray.mean())
        # Edge/texture
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        edge_var = float(lap.var())

        # Motion (MAD tra frame)
        motion = 0.0
        if prev_gray is not None:
            motion = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
        prev_gray = gray

        # Timestamp stimato
        ts = (grabbed / (src_fps if src_fps > 0 else fps)) if src_fps > 0 else (len(frames_info) / fps)

        frames_info.append({
            "t": ts,
            "y_mean": y_mean,
            "edge_var": edge_var,
            "motion": motion
        })
        total_frames += 1

        # Bound extra sicurezza su numero frame totali
        if total_frames >= int(fps * max_seconds) + 90:
            break

    cap.release()

    # Binning per secondo
    if not frames_info:
        return {
            "error": "no_frames_sampled",
            "width": width, "height": height, "src_fps": src_fps, "duration": duration
        }

    # raggruppa per int(t)
    bins = {}
    for f in frames_info:
        sec = int(f["t"])
        arr = bins.setdefault(sec, {"y_mean": [], "edge_var": [], "motion": []})
        arr["y_mean"].append(f["y_mean"])
        arr["edge_var"].append(f["edge_var"])
        arr["motion"].append(f["motion"])

    timeline = []
    for sec in sorted(bins.keys()):
        arr = bins[sec]
        timeline.append({
            "t": sec,
            "y_mean": float(np.mean(arr["y_mean"])) if arr["y_mean"] else 0.0,
            "edge_var": float(np.mean(arr["edge_var"])) if arr["edge_var"] else 0.0,
            "motion": float(np.mean(arr["motion"])) if arr["motion"] else 0.0,
        })

    # Statistiche globali
    y_means = np.array([x["y_mean"] for x in timeline], dtype=np.float32)
    edge_means = np.array([x["edge_var"] for x in timeline], dtype=np.float32)
    motion_means = np.array([x["motion"] for x in timeline], dtype=np.float32)

    stats = {
        "width": width,
        "height": height,
        "src_fps": src_fps,
        "duration": duration,
        "sampled_fps": fps if src_fps > 0 else fps,  # logico
        "frames_sampled": int(total_frames),
        "timeline": timeline,
        "summary": {
            "y_mean_avg": float(y_means.mean()) if y_means.size else 0.0,
            "edge_var_avg": float(edge_means.mean()) if edge_means.size else 0.0,
            "motion_avg": float(motion_means.mean()) if motion_means.size else 0.0,
            "motion_std": float(motion_means.std()) if motion_means.size else 0.0,
        }
    }
    return stats
