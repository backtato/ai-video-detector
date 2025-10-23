from typing import Dict, Any, Optional

# Euristica robusta e conservativa: segna "social" se codec/durata/bitrate tipici
def detect_source_profile(ffp: Dict[str, Any], force_social: Optional[bool] = None) -> str:
    if force_social is True:
        return "social"
    if force_social is False:
        return "clean"

    try:
        fmt = ffp.get("format", {}) or {}
        duration_s = float(fmt.get("duration", "0") or "0")
    except Exception:
        duration_s = 0.0

    # Bitrate: molti social lo nascondono; se manca non lo usiamo
    try:
        bit_rate = int(fmt.get("bit_rate", "0") or "0")
    except Exception:
        bit_rate = 0

    codecs = set()
    for s in ffp.get("streams", []):
        cn = (s.get("codec_name") or "").lower()
        ct = (s.get("codec_type") or "").lower()
        if ct == "video" and cn:
            codecs.add(cn)

    # Regole semplici:
    # - AV1/H.264 su clip corti (<= 12s) = probabile social
    # - bitrate molto basso (< 600 kbps) = probabile social
    # - mancanza completa di metadati "ricchi" = indizio social (non forzante)
    if ("av1" in codecs or "h264" in codecs) and duration_s <= 12.0:
        return "social"
    if bit_rate and bit_rate < 600_000:
        return "social"

    # fallback
    return "clean"
