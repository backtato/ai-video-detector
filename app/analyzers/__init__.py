import os
from typing import Optional, Dict, Any, List
from .sampler import sample_frames, sample_audio_wav, ffprobe_meta
from .features_video import video_frame_scores
from .features_audio import audio_scores
from .forensic import forensic_checks
from .fusion import fuse_and_label

def run(path: str, source_url: Optional[str]=None, resolved_url: Optional[str]=None, ffprobe_json: Optional[Dict[str,Any]]=None) -> Dict[str,Any]:
    meta = ffprobe_meta(ffprobe_json)
    forensic = forensic_checks(ffprobe_json or {})
    frames_dir, times = sample_frames(
        path,
        target_frames=36,
        prefer_seconds=min(meta.get("duration", 10.0), 30.0)
    )
    wav_path = sample_audio_wav(path, sr=16000)
    v_scores, v_timeline = video_frame_scores(frames_dir, times)
    a_scores, a_timeline = audio_scores(wav_path, meta.get("duration", 0.0), window_sec=2.0)
    out = fuse_and_label(meta, forensic, v_scores, v_timeline, a_scores, a_timeline)
    out["meta"]["source_url"] = source_url
    out["meta"]["resolved_url"] = resolved_url
    try:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
    except: pass
    return out
