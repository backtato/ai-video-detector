import cv2
import numpy as np
from typing import List

def extract_frames(path: str, max_frames: int = 64, stride: int = 5) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames

def video_duration_fps(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    duration = (frame_count / fps) if fps > 0 else 0.0
    return duration, fps, int(frame_count)
