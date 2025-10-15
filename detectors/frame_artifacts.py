import cv2, numpy as np

def _hf(gray):
    # Varianza del Laplaciano → asimmetrica, normalizzata con tanh
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var())
    return np.tanh(var / 1000.0)

def _blk(gray):
    """
    Blockiness robusta: confronta i bordi "8x8" prendendo differenze tra
    colonne (c-1, c) per c in {8,16,...} e righe (r-1, r) per r in {8,16,...}.
    Evita slicing con shape diverse e broadcasting.
    """
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0

    g = gray.astype(np.float32)

    # colonne e righe candidate agli spigoli di blocco
    cols = list(range(8, w, 8))
    rows = list(range(8, h, 8))

    v_edges = []
    for c in cols:
        # diff colonna bordo: c-1 vs c
        # (tutte le righe, stessa lunghezza)
        d = np.abs(g[:, c].reshape(-1) - g[:, c-1].reshape(-1))
        v_edges.append(float(d.mean()))

    h_edges = []
    for r in rows:
        # diff riga bordo: r-1 vs r
        d = np.abs(g[r, :].reshape(-1) - g[r-1, :].reshape(-1))
        h_edges.append(float(d.mean()))

    v_mean = float(np.mean(v_edges)) if v_edges else 0.0
    h_mean = float(np.mean(h_edges)) if h_edges else 0.0
    m = v_mean + h_mean

    # squash
    return np.tanh(m / 20.0)

def _noise_consistency(frames_gray):
    if len(frames_gray) < 3:
        return 0.5
    diffs = []
    for i in range(1, len(frames_gray)):
        a = frames_gray[i].astype(np.float32)
        b = frames_gray[i-1].astype(np.float32)
        diffs.append(float(np.mean(np.abs(a - b))))
    v = np.var(diffs) if len(diffs) > 1 else 0.0
    # più varianza → meno consistente → più “sospetto”; invertiamo e squash
    return 1.0 - np.tanh(v / 50.0)

def score_frame_artifacts(frames_bgr):
    if not frames_bgr:
        return {"score": 0.6, "notes": ["No frames"]}

    hf_vals = []
    blk_vals = []
    gstack = []

    for f in frames_bgr:
        try:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        except Exception:
            # frame corrotto o vuoto
            continue
        gstack.append(gray)
        hf_vals.append(_hf(gray))
        blk_vals.append(_blk(gray))

    if not hf_vals or not blk_vals:
        return {"score": 0.6, "notes": ["Invalid/empty frames"]}

    hf = float(np.mean(hf_vals))
    blk = float(np.mean(blk_vals))
    noise = _noise_consistency(gstack)

    # combinazione semplice (euristica MVP)
    raw = 0.4 * blk + 0.3 * (1.0 - noise) + 0.3 * (1.0 - abs(hf - 0.6))
    score = max(0.0, min(1.0, raw))

    return {
        "score": score,
        "notes": [f"HF~{hf:.2f}", f"Blockiness~{blk:.2f}", f"NoiseConsistency~{noise:.2f}"]
    }
