import subprocess
import json
from typing import Dict, Any

def _probe_audio_streams(meta: Dict[str, Any]) -> int:
    c = 0
    for st in meta.get("streams", []):
        if st.get("codec_type") == "audio":
            c += 1
    return c

def score_audio(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightweight audio signal proxy using metadata only (robust, fast).
    Heuristics:
      - If there's audio: start near-neutral (0.45).
      - If codec is typical of screen-captures (opus/webm) or TTS-only tracks → slightly higher.
    """
    audio_streams = [s for s in meta.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio_streams:
        return {"score": 0.6, "details": {"audio_present": False, "hint": "no_audio"}}

    # default baseline
    score = 0.45
    codecs = {s.get("codec_name", "") for s in audio_streams}
    sample_rates = {s.get("sample_rate", "") for s in audio_streams}

    if "opus" in codecs:  # common on webm/screen records
        score += 0.05
    if "aac" in codecs:
        score -= 0.02
    if "pcm_s16le" in codecs:
        score -= 0.02

    # Unusual sample rates may nudge suspicion
    if any(sr in ("8000", "11025") for sr in sample_rates):
        score += 0.05

    score = max(0.0, min(1.0, score))
    return {"score": score, "details": {"audio_present": True, "codecs": list(codecs), "sample_rates": list(sample_rates)}}
