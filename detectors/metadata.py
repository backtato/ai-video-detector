import subprocess, json
def ffprobe(path):
    try:
        out = subprocess.check_output([
            "ffprobe","-v","error","-show_entries",
            "format=format_name,format_long_name,duration,bit_rate",
            "-show_streams","-print_format","json", path
        ])
        return json.loads(out.decode("utf-8"))
    except Exception:
        return {}

def score_metadata(info):
    score=0.5; notes=[]
    try:
        fmt = info.get("format", {})
        duration = float(fmt.get("duration",0) or 0)
        bit_rate = float(fmt.get("bit_rate",0) or 0)
        format_name = fmt.get("format_name","")
        if duration==0: score+=0.1; notes.append("Missing/zero duration")
        if bit_rate==0: score+=0.1; notes.append("Missing/zero bitrate")
        if "mp4" not in format_name and "mov" not in format_name and "matroska" not in format_name:
            score+=0.05; notes.append(f"Uncommon container {format_name}")
        streams = info.get("streams",[])
        has_video = any(s.get("codec_type")=="video" for s in streams)
        has_audio = any(s.get("codec_type")=="audio" for s in streams)
        if not has_audio: score+=0.05; notes.append("No audio stream")
        if not has_video: score+=0.3; notes.append("No video stream")
        score = max(0.0, min(1.0, score))
    except Exception as e:
        notes.append(f"metadata parse error: {e}"); score=0.6
    return {"score": score, "notes": notes}
