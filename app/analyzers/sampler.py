import os, tempfile, subprocess, json, glob
from typing import Tuple, List, Dict, Any

def _run(cmd: list):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)

def ffprobe_meta(ffprobe_json: Dict[str,Any]) -> Dict[str,Any]:
    fmt = (ffprobe_json or {}).get("format", {}) or {}
    streams = (ffprobe_json or {}).get("streams", []) or []
    v = next((s for s in streams if s.get("codec_type")=="video"), {})
    a = next((s for s in streams if s.get("codec_type")=="audio"), {})
    def _fps(s):
        try:
            r = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/1"
            n,d = r.split("/")
            return float(n)/float(d) if float(d)!=0 else 0.0
        except: return 0.0
    return {
        "width": v.get("width"),
        "height": v.get("height"),
        "fps": _fps(v),
        "duration": float(fmt.get("duration") or 0.0),
        "bit_rate": float(fmt.get("bit_rate") or 0.0),
        "vcodec": v.get("codec_name"),
        "acodec": a.get("codec_name"),
        "format_name": fmt.get("format_name"),
    }

def sample_frames(path: str, target_frames: int=24, prefer_seconds: float=10.0) -> Tuple[str, List[float]]:
    outdir = tempfile.mkdtemp(prefix="frames_")
    info = _run(["ffprobe","-v","error","-hide_banner","-show_entries","format=duration","-of","json",path])
    dur = 0.0
    try:
        dur = float((json.loads(info.stdout or "{}").get("format") or {}).get("duration") or 0.0)
    except: pass
    span = min(dur, prefer_seconds) if dur>0 else prefer_seconds
    if target_frames <= 0: target_frames = 12
    fps = max(target_frames / max(span, 0.1), 0.5)
    cmd = [
        "ffmpeg","-hide_banner","-v","error",
        "-ss","0","-t",f"{span:.3f}",
        "-i", path,
        "-vf", f"fps={fps:.4f}",
        "-qscale:v","2",
        os.path.join(outdir, "frame_%05d.jpg")
    ]
    _run(cmd)
    times = [i*(1.0/fps) for i in range(1, target_frames+1)]
    files = sorted(glob.glob(os.path.join(outdir,"frame_*.jpg")))
    if files and len(files) < len(times): times = times[:len(files)]
    if not files: times = []
    return outdir, times

def sample_audio_wav(path: str, sr: int=16000) -> str:
    wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav_path = wav.name
    wav.close()
    cmd = ["ffmpeg","-hide_banner","-v","error","-i", path, "-ac","1","-ar",str(sr),"-f","wav", wav_path]
    res = _run(cmd)
    if res.returncode != 0:
        open(wav_path,"wb").close()
    return wav_path