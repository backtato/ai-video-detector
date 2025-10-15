import cv2
import numpy as np
from typing import List, Tuple

def _hf(gray: np.ndarray) -> float:
    """
    Alta frequenza: varianza del Laplaciano → squash con tanh.
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var())
    return float(np.tanh(var / 1000.0))

def _blk(gray: np.ndarray) -> float:
    """
    Blockiness robusta su griglia 8x8: differenze tra (c-1, c) e (r-1, r).
    Evita broadcasting inconsistente su shape diverse.
    """
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0

    g = gray.astype(np.float32, copy=False)

    cols = list(range(8, w, 8))
    rows = list(range(8, h, 8))

    v_edges: List[float] = []
    for c in cols:
        d = np.abs(g[:, c] - g[:, c - 1])
        v_edges.append(float(d.mean()))

    h_edges: List[float] = []
    for r in rows:
        d = np.abs(g[r, :] - g[r - 1, :])
        h_edges.append(float(d.mean()))

    v_mean = float(np.mean(v_edges)) if v_edges else 0.0
    h_mean = float(np.mean(h_edges)) if h_edges else 0.0
    m = v_mean + h_mean

    return float(np.tanh(m / 20.0))

def _noise_consistency(frames_gray: List[np.ndarray]) -> float:
    """
    Coerenza del rumore frame-to-frame: più varianza nelle differenze → meno consistente.
    Ritorna [0..1], dove 1 = molto consistente (meno sospetto).
    """
    if len(frames_gray) < 3:
        return 0.5
    diffs: List[float] = []
    prev = frames_gray[0].astype(np.float32, copy=False)
    for i in range(1, len(frames_gray)):
        cur = frames_gray[i].astype(np.float32, copy=False)
        diffs.append(float(np.mean(np.abs(cur - prev))))
        prev = cur
    v = np.var(diffs) if len(diffs) > 1 else 0.0
    return float(1.0 - np.tanh(v / 50.0))

def _uniform_indices(total_frames: int, target_frames: int) -> List[int]:
    """
    Seleziona ~target_frames indici uniformi nell'intervallo [0, total_frames).
    """
    if target_frames <= 0 or total_frames <= 0:
        return []
    if target_frames >= total_frames:
        return list(range(total_frames))
    step = max(1, total_frames // target_frames)
    idxs = list(range(0, total_frames, step))[:target_frames]
    # Garantiamo che lo 0 e l'ultimo (se possibile) siano inclusi
    if idxs and idxs[0] != 0:
        idxs[0] = 0
    if idxs and idxs[-1] != total_frames - 1:
        idxs[-1] = min(total_frames - 1, idxs[-1])
    return idxs

def _read_gray_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray | None:
    """
    Posiziona il cursore al frame index e legge un frame gray.
    Se fallisce il read, prova ±1 frame intorno (tolleranza piccola); se ancora fallisce, None.
    """
    # tentativo principale
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = cap.read()
    if ok and bgr is not None:
        try:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None

    # micro-riprove locali (senza resettare a 0)
    for off in (-1, 1, -2, 2):
        j = index + off
        if j < 0:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, j)
        ok2, bgr2 = cap.read()
        if ok2 and bgr2 is not None:
            try:
                return cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)
            except Exception:
                return None
    return None

def score_frame_artifacts(path: str, target_frames: int = 120) -> dict:
    """
    Apre il video da `path`, estrae ~target_frames a passo uniforme e calcola:
      - HF (Laplaciano)
      - Blockiness 8x8
      - Noise consistency (frame-to-frame)
    Ritorna:
      {
        "score": float in [0..1],
        "frames_analyzed": int,
        "notes": ["HF~x.xx", "Blockiness~x.xx", "NoiseConsistency~x.xx"]
      }
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV cannot open video")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Se CAP_PROP_FRAME_COUNT non è affidabile, tentativo di scan rapido limitato
        if total <= 0:
            # fallback: leggi fino a 1000 step a distanza fissa per stimare
            # (manteniamolo snello; in MVP, spesso FFprobe dà comunque il count)
            scan = 0
            pos = 0
            step = 5
            while scan < 1000:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ok, _ = cap.read()
                if not ok:
                    break
                scan += 1
                pos += step
            total = max(total, scan * step)

        idxs = _uniform_indices(total, target_frames)
        if not idxs:
            # Nessun frame: score neutro ma informativo
            return {"score": 0.6, "frames_analyzed": 0, "notes": ["No frames"]}

        hf_vals: List[float] = []
        blk_vals: List[float] = []
        gstack: List[np.ndarray] = []

        for i in idxs:
            gray = _read_gray_frame(cap, i)
            if gray is None:
                continue
            gstack.append(gray)
            hf_vals.append(_hf(gray))
            blk_vals.append(_blk(gray))

        analyzed = len(gstack)
        if analyzed == 0:
            return {"score": 0.6, "frames_analyzed": 0, "notes": ["Invalid/empty frames"]}

        hf = float(np.mean(hf_vals)) if hf_vals else 0.5
        blk = float(np.mean(blk_vals)) if blk_vals else 0.5
        noise = _noise_consistency(gstack)

        # combinazione semplice (euristica MVP) — stessa tua logica
        raw = 0.4 * blk + 0.3 * (1.0 - noise) + 0.3 * (1.0 - abs(hf - 0.6))
        score = float(max(0.0, min(1.0, raw)))

        return {
            "score": score,
            "frames_analyzed": analyzed,
            "notes": [f"HF~{hf:.2f}", f"Blockiness~{blk:.2f}", f"NoiseConsistency~{noise:.2f}"],
        }
    finally:
        cap.release()
