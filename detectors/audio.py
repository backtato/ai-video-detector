def score_audio(info):
    streams = info.get("streams", [])
    audio_streams = [s for s in streams if s.get("codec_type")=="audio"]
    if not audio_streams:
        return {"score":0.55, "notes":["No audio stream"]}
    return {"score":0.45, "notes":["Audio present (placeholder)"]}

