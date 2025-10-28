# app/analyzers/heuristics_v2.py
# Heuristics estese: usa nuovi summary video, bitrate/meta, e inferenze conservative.
# Mantiene compatibilità: compute_hints(video_stats, audio_stats, meta) → dict

def _get(meta: dict, *path, default=None):
    cur = meta or {}
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default

def compute_hints(video_stats: dict = None, audio_stats: dict = None, meta: dict = None) -> dict:
    hints = {}
    v = video_stats or {}
    a = audio_stats or {}
    m = meta or {}

    # Bitrate / formato
    bit_rate = float(_get(m, "bit_rate", default=0.0) or _get(m, "summary", "bit_rate", default=0.0) or 0.0)
    width = float(_get(v, "width", default=0.0))
    height = float(_get(v, "height", default=0.0))
    duration = float(_get(v, "duration", default=0.0))

    if bit_rate and bit_rate < 400_000:
        hints["very_low_bitrate"] = {"score": 0.65, "reason": "Bitrate molto basso."}
    elif bit_rate and bit_rate < 800_000:
        hints["low_bitrate"] = {"score": 0.58, "reason": "Bitrate basso."}

    # Video summary
    vsum = v.get("summary", {})
    motion_avg = float(vsum.get("motion_avg", 0.0))
    edge_avg = float(vsum.get("edge_var_avg", 0.0))
    dup_avg = float(vsum.get("dup_avg", 0.0))
    block_avg = float(vsum.get("blockiness_avg", 0.0))
    band_avg = float(vsum.get("banding_avg", 0.0))
    optflow_avg = float(vsum.get("optflow_mag_avg", 0.0))

    # Bassi motion & texture
    if motion_avg < 0.15:
        hints["low_motion"] = {"score": 0.58, "reason": "Dinamica ridotta."}
    if edge_avg < 20.0:  # soglia empirica varianza Laplacian bassa
        hints["low_texture"] = {"score": 0.56, "reason": "Bassa definizione dettagli/edge."}
    if "low_motion" in hints and "low_texture" in hints:
        hints["low_motion_low_texture"] = {"score": 0.62, "reason": "Pochi dettagli e bassa dinamica."}

    # Dup frames + motion basso → screen/interpolazioni/AI
    if dup_avg > 0.7 and motion_avg < 0.5:
        hints["many_duplicates"] = {"score": 0.66, "reason": "Molti frame duplicati con bassa dinamica."}

    # Compressione aggressiva
    if block_avg > 0.35 or band_avg > 0.35:
        hints["heavy_compression"] = {"score": 0.62, "reason": "Forte compressione (blockiness/banding)."}

    # Flow-motion mismatch
    if (optflow_avg < 0.05 and motion_avg > 0.3) or (optflow_avg > 1.5 and motion_avg < 0.2):
        hints["flow_motion_mismatch"] = {"score": 0.64, "reason": "Ottico vs motion incongruente."}

    # Screen-capture probabile se (bassa dinamica) + (bitrate basso o heavy compression)
    if (("low_motion_low_texture" in hints) or ("low_motion" in hints)) and \
       (("low_bitrate" in hints) or ("very_low_bitrate" in hints) or ("heavy_compression" in hints)):
        hints["likely_screen_capture"] = {"score": 0.70, "reason": "Pattern compatibile con registrazione schermo/ricompressione."}

    # Audio flags (non spingono molto, ma aggiungono contesto)
    aflags = (a or {}).get("flags_audio", []) or []
    if "very_low_energy" in aflags:
        hints["audio_low_energy"] = {"score": 0.55, "reason": "Audio a bassa energia."}
    if "flat_spectrum" in aflags:
        hints["audio_flat_spectrum"] = {"score": 0.57, "reason": "Spettro appiattito (compressione/TTS)."}
    if "tts_like" in aflags:
        hints["audio_tts_like"] = {"score": 0.66, "reason": "Pattern vocale simile a TTS."}

    # Forensics: C2PA → lievemente verso 'real' (conservativo), assenza neutra.
    c2pa_present = bool(_get(m, "forensic", "c2pa", "present", default=False))
    if c2pa_present:
        hints["c2pa_present"] = {"score": 0.42, "reason": "C2PA/JUMBF presente (indicazione pro-reale)."}

    # Aspetto e AR 'strani' (molto grezzo, solo suggerimento)
    if width and height:
        ar = width / max(height, 1.0)
        if ar < 0.9 or ar > 2.2:
            hints["unusual_aspect_ratio"] = {"score": 0.52, "reason": "Aspect ratio inusuale."}

    return hints
