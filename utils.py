import cv2
def extract_frames(path, max_frames=64, stride=5):
    cap=cv2.VideoCapture(path); frames=[]; idx=0
    while True:
        ret,frame=cap.read()
        if not ret: break
        if idx % stride == 0:
            frames.append(frame)
            if len(frames)>=max_frames: break
        idx+=1
    cap.release(); return frames
def video_duration_fps(path):
    cap=cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    fc  = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    dur=(fc/fps) if fps>0 else 0.0
    return dur, fps, int(fc)
