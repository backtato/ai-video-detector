import cv2
import numpy as np

def _frame_hash(img):
    x = cv2.resize(img, (32, 32))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    return cv2.img_hash.AverageHash_create().compute(x)

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
    prev = None
    dup = 0
    total = 0
    flow_means = []
    flow_vars = []
    textures = []
    timeline_ai = []

    index = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if index % step == 0:
            ok, frame = cap.retrieve()
            if not ok: break
            total += 1
            # duplicate detection
            hsh = _frame_hash(frame)
            if prev is not None and (hsh == prev).all():
                dup += 1
            prev = hsh
            # flow vs previous sampled frame (use grayscale small)
            if 'prev_frame' in locals():
                a = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                b = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                a = cv2.resize(a, (320, 320)); b = cv2.resize(b, (320, 320))
                flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
                flow_means.append(float(np.mean(mag)))
                flow_vars.append(float(np.var(mag)))
            prev_frame = frame
            # texture flatness via Laplacian variance (low => flat)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            textures.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            # simple per-sample ai suspicion: very flat + low motion → higher AI suspicion
            ai_susp = float(np.clip(1.0 - (textures[-1]/(textures[-1]+1000.0)) * (1.0 + (flow_means[-1] if flow_means else 0.0)), 0.0, 1.0))
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
    # align timeline length to seconds
    tlen = int(max(1, round(duration)))
    if len(timeline_ai) < tlen:
        # simple repeat to fill
        if timeline_ai:
            last = timeline_ai[-1]
            timeline_ai += [last]*(tlen-len(timeline_ai))
        else:
            timeline_ai = [0.5]*tlen
    else:
        timeline_ai = timeline_ai[:tlen]
    return {"timeline": timeline_ai, "summary": summary, "timeline_ai": timeline_ai}