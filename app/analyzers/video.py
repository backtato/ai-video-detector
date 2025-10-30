import cv2
import numpy as np

def _average_hash(img, size=32):
    """
    Implementazione leggera di Average Hash:
    - resize grayscale a NxN
    - media dei pixel
    - ritorna un vettore booleano (flatten)
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    mean = g.mean()
    return (g >= mean).astype(np.uint8).flatten()

def analyze(path: str, meta: dict):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"timeline": [], "summary": {}, "timeline_ai": []}
    fps = meta.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 0.0
    w = meta.get("width") or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = meta.get("height") or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = meta.get("duration") or (cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps>0 else 0.0)

    # sample ~2 fps
    step = max(1, int(round((fps or 30)/2)))
    prev_hash = None
    dup = 0
    total = 0
    flow_means = []
    flow_vars = []
    textures = []
    timeline_ai = []

    index = 0
    prev_frame_small = None
    while True:
        ret = cap.grab()
        if not ret:
            break
        if index % step == 0:
            ok, frame = cap.retrieve()
            if not ok: break
            total += 1

            # duplicate detection via avg-hash
            hsh = _average_hash(frame, size=32)
            if prev_hash is not None:
                # Hamming distance; se 0 => duplicato perfetto
                ham = int(np.sum(hsh ^ prev_hash))
                if ham == 0:
                    dup += 1
            prev_hash = hsh

            # optical flow (Farneback) su frames ridotti
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 320))
            if prev_frame_small is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_frame_small, small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
                flow_means.append(float(np.mean(mag)))
                flow_vars.append(float(np.var(mag)))
            prev_frame_small = small

            # texture flatness via var(Laplacian)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            textures.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

            # semplice indice di "AI-suspicion": bassa texture + bassa motion → più alto
            tex = textures[-1]
            mot = flow_means[-1] if flow_means else 0.0
            ai_susp = float(np.clip(1.0 - (tex/(tex+1000.0)) * (1.0 + mot), 0.0, 1.0))
            timeline_ai.append(ai_susp)
        index += 1
    cap.release()

    dup_density = float(dup / max(1, total-1))
    sc_rate = float(np.mean(np.array(flow_vars)>0.5)) if flow_vars else 0.0

    summary = {
        "dup_density": dup_density,
        "scene_change_rate": sc_rate,
        "flow_mean": float(np.mean(flow_means)) if flow_means else 0.0,
        "flow_var": float(np.var(flow_means)) if flow_means else 0.0,
        "texture_var": float(np.var(textures)) if textures else 0.0,
        "w": int(w), "h": int(h), "fps": float(fps)
    }

    # allinea la timeline ai secondi
    tlen = int(max(1, round(duration)))
    if len(timeline_ai) < tlen:
        if timeline_ai:
            last = timeline_ai[-1]
            timeline_ai += [last]*(tlen-len(timeline_ai))
        else:
            timeline_ai = [0.5]*tlen
    else:
        timeline_ai = timeline_ai[:tlen]

    return {"timeline": timeline_ai, "summary": summary, "timeline_ai": timeline_ai}