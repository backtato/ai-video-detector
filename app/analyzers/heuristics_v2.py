import math

def _safe_get(d, *keys, default=None):
    cur = d
    try:
        for k in keys:
            if cur is None:
                return default
            cur = cur.get(k, None)
        return cur if cur is not None else default
    except Exception:
        return default

def compute_hints(meta: dict, video_stats: dict, audio_stats: dict) -> dict:
    """
    Restituisce una mappa di hints, ciascuno con:
      { "score": float[0..1], "reason": str }

    Convenzione: score > 0.5 spinge verso "AI" (negativo),
                 score < 0.5 spinge verso "REAL" (positivo).
    """
    hints = {}

    # --- Meta quicktime/bitrate/aspect ---
    width = _safe_get(video_stats, "width", default=0) or _safe_get(meta, "summary", "width", default=0) or 0
    height = _safe_get(video_stats, "height", default=0) or _safe_get(meta, "summary", "height", default=0) or 0
    bit_rate = float(_safe_get(meta, "summary", "bit_rate", default=0) or 0)
    duration = float(_safe_get(video_stats, "duration", default=0.0) or _safe_get(meta, "summary", "duration", default=0.0) or 0.0)

    if width and height:
        ar = float(width) / float(height)
        # se aspect ratio molto strano
        if ar < 0.5 or ar > 3.0:
            hints["aspect_inconsistent"] = {
                "score": 0.7,
                "reason": f"Aspect ratio atipico ({ar:.2f})."
            }

    # bitrate basso + alta compressione spesso accompagna schermo/ricompressioni social
    if bit_rate > 0:
        if bit_rate < 800_000:  # <0.8 Mbps
            hints["very_low_bitrate"] = {"score": 0.65, "reason": f"Bitrate molto basso ({bit_rate:.0f})."}
        elif bit_rate < 2_000_000:
            hints["low_bitrate"] = {"score": 0.58, "reason": f"Bitrate relativamente basso ({bit_rate:.0f})."}
        else:
            hints["adequate_bitrate"] = {"score": 0.45, "reason": f"Bitrate adeguato ({bit_rate:.0f})."}

    # --- Video dynamics ---
    v_summary = _safe_get(video_stats, "summary", default={}) or {}
    motion_avg = float(v_summary.get("motion_avg", 0.0))
    motion_std = float(v_summary.get("motion_std", 0.0))
    edge_avg = float(v_summary.get("edge_var_avg", 0.0))

    # soglie euristiche (dipendono dalla scala usata in video.py — qui conservative)
    if motion_avg < 0.5 and edge_avg < 100.0:
        hints["low_motion_low_texture"] = {
            "score": 0.68,
            "reason": "Bassa dinamica e bassa texture media: possibile schermo o contenuto sintetico."
        }
    elif motion_avg < 0.5:
        hints["low_motion"] = {"score": 0.60, "reason": "Bassa dinamica rilevata."}
    elif edge_avg < 100.0:
        hints["low_texture"] = {"score": 0.58, "reason": "Bassa texture media (edge var)."}
    else:
        hints["motion_texture_ok"] = {"score": 0.45, "reason": "Dinamica/texture compatibili con cattura reale."}

    # alta variabilità motion → può indicare real
    if motion_std > 1.5:
        hints["motion_variability_ok"] = {"score": 0.42, "reason": "Variabilità di movimento buona."}

    # --- Audio energy ---
    a_tl = _safe_get(audio_stats, "timeline", default=[]) or []
    if a_tl:
        rms_vals = [float(x.get("rms", 0.0)) for x in a_tl]
        zcr_vals = [float(x.get("zcr", 0.0)) for x in a_tl]
        if rms_vals:
            rms_mean = sum(rms_vals) / len(rms_vals)
            if rms_mean < 0.01:
                hints["very_low_audio_energy"] = {"score": 0.62, "reason": "Energia audio molto bassa/silenzio esteso."}
            else:
                hints["audio_energy_ok"] = {"score": 0.47, "reason": "Energia audio presente."}
        if zcr_vals:
            zcr_mean = sum(zcr_vals) / len(zcr_vals)
            # zcr molto basso e rms basso → sintetico/silenzio
            if zcr_mean < 0.02 and (hints.get("very_low_audio_energy") or rms_vals and (sum(rms_vals)/len(rms_vals) < 0.01)):
                hints["flat_audio"] = {"score": 0.60, "reason": "Audio piatto: pochi passaggi/variazioni."}

    # --- Forensic flags (C2PA & QuickTime tags) se presenti in meta ---
    forensic = _safe_get(meta, "forensic", default={}) or {}
    c2pa_present = bool(_safe_get(forensic, "c2pa", "present", default=False))
    if c2pa_present:
        # Presenza C2PA non garantisce real, ma è un indizio positivo
        hints["c2pa_present"] = {"score": 0.40, "reason": "C2PA presente (firma digitale metadati)."}
    else:
        hints["c2pa_absent"] = {"score": 0.52, "reason": "C2PA assente (comune, non determinante)."}

    # --- Likely screen capture (grossolana) ---
    # bassa motion, bassa texture, bitrate basso → probabile schermo
    if ("low_motion_low_texture" in hints or "low_motion" in hints) and \
       ("low_bitrate" in hints or "very_low_bitrate" in hints):
        hints["likely_screen_capture"] = {"score": 0.70, "reason": "Pattern compatibile con registrazione di schermo/ricompressione forte."}

    return hints
