import subprocess, json, math
import cv2
import numpy as np

def _ffprobe_json(path: str) -> dict:
    try:
        out = subprocess.check_output([
            "ffprobe","-v","error",
            "-select_streams","v:0",
            "-show_entries","format=duration:stream=avg_frame_rate,r_frame_rate,nb_frames",
            "-print_format","json", path
        ])
        return json.loads(out.decode("utf-8"))
    except Exception:
        return {}

def _parse_fps(s: str) -> float:
    """
    Converte "30000/1001" → 29.97. Restituisce 0.0 su input non valido.
    """
    if not s:
        return 0.0
    if "/" in s:
        try:
            a,b = s.split("/")
            a = float(a or 0); b = float(b or 1)
            return 0.0 if b == 0 else a/b
        except Exception:
            return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0

def video_duration_fps(path: str):
    """
    Ritorna (duration_sec, fps, frame_count) con stime robuste:
    - Prova ffprobe (preferito)
    - Fallback OpenCV
    - Clampa fps in [1, 120]
    - Se frame_count implausibile, ricalcola come round(duration*fps)
    """
    info = _ffprobe_json(path)
    dur = 0.0
    fps = 0.0
    nbf = 0

    # durata
    try:
        dur = float(info.get("format", {}).get("duration", 0.0) or 0.0)
    except Exception:
        dur = 0.0

    # fps
    streams = info.get("streams") or []
    if streams:
        st = streams[0]
        fps = _parse_fps(st.get("avg_frame_rate") or "") or _parse_fps(st.get("r_frame_rate") or  "")
        try:
            nbf = int(st.get("nb_frames") or 0)
        except Exception:
            nbf = 0

    # Fallback OpenCV
    if dur <= 0.0 or fps <= 0.0:
        cap = cv2.VideoCapture(path)
        if dur <= 0.0:
            fc  = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            cfps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            if cfps > 0:
                dur = (fc / cfps) if fc > 0 else 0.0
        if fps <= 0.0:
            cfps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            if cfps > 0:
                fps = float(cfps)
        cap.release()

    # clamp fps in range sensato
    if fps <= 0.0:
        fps = 24.0
    fps = max(1.0, min(120.0, float(fps)))

    # frame_count coerente
    if nbf <= 0:
        if dur > 0.0:
            nbf = int(round(dur * fps))
        else:
            # fallback estremo: usa OpenCV
            cap = cv2.VideoCapture(path)
            nbf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()

    return float(dur), float(fps), int(nbf)

def extract_frames_evenly(path, max_frames=64):
    """
    Campionamento UNIFORME su tutta la durata: evita stride derivati da FPS inaccurati.
    """
    dur, fps, nbf = video_duration_fps(path)
    if nbf <= 0:
        return []

    # quante estrazioni effettive:
    k = int(min(max_frames, nbf))
    if k <= 0:
        return []

    indices = np.linspace(0, nbf-1, num=k, dtype=int)
    frames = []

    cap = cv2.VideoCapture(path)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            # tenta una lettura sequenziale
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok2, frame2 = cap.read()
            if ok2:
                frames.append(frame2)
            continue
        frames.append(frame)
    cap.release()
    return frames
