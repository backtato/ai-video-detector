import cv2, numpy as np

def _hf(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F); var = float(lap.var())
    return np.tanh(var/1000.0)

def _blk(gray):
    h,w = gray.shape
    v = gray[:,7::8].astype(np.float32) - gray[:,8::8].astype(np.float32) if w>16 else np.zeros((1,1),np.float32)
    hE= gray[7::8,:].astype(np.float32) - gray[8::8,:].astype(np.float32) if h>16 else np.zeros((1,1),np.float32)
    m = float(np.mean(np.abs(v))) + float(np.mean(np.abs(hE)))
    return np.tanh(m/20.0)

def _noise(frames_gray):
    if len(frames_gray)<3: return 0.5
    diffs=[]
    for i in range(1,len(frames_gray)):
        diffs.append(float(np.mean(np.abs(frames_gray[i].astype(np.float32)-frames_gray[i-1].astype(np.float32)))))
    v = np.var(diffs) if len(diffs)>1 else 0.0
    return 1.0 - np.tanh(v/50.0)

def score_frame_artifacts(frames_bgr):
    if not frames_bgr: return {"score":0.6,"notes":["No frames"]}
    hf_vals=[]; blk_vals=[]; g=[]
    for f in frames_bgr:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); g.append(gray)
        hf_vals.append(_hf(gray)); blk_vals.append(_blk(gray))
    hf=float(np.mean(hf_vals)) if hf_vals else 0.0
    blk=float(np.mean(blk_vals)) if blk_vals else 0.0
    noise=_noise(g)
    raw=0.4*blk + 0.3*(1.0-noise) + 0.3*(1.0-abs(hf-0.6))
    return {"score": max(0.0,min(1.0,raw)), "notes":[f"HF~{hf:.2f}", f"Blockiness~{blk:.2f}", f"NoiseConsistency~{noise:.2f}"]}
