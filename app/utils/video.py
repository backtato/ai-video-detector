
import cv2 as cv
import numpy as np
from typing import Tuple, List

def read_video_frames(path: str, target_fps: int = 6, max_frames: int = 480) -> Tuple[List[np.ndarray], float, int, int]:
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Impossibile aprire il video.")

    src_fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    step = max(int(round(src_fps / max(target_fps,1))), 1)
    frames = []
    i = 0
    kept = 0
    while kept < max_frames:
        ret = cap.grab()
        if not ret:
            break
        if i % step == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break
            if max(height, width) > 1080:
                r = 1080 / max(height, width)
                new_w, new_h = int(width*r), int(height*r)
                frame = cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)
            frames.append(frame)
            kept += 1
        i += 1

    cap.release()
    effective_fps = min(src_fps, float(target_fps))
    return frames, effective_fps, width, height
