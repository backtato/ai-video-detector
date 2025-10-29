from typing import Dict, Any, List, Tuple

THRESH_REAL = 0.35
THRESH_AI   = 0.72

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))

def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def _bpp(meta: Dict[str, Any], vstats: Dict[str, Any]) -> float:
    br  = float(_safe_get(meta, ["bit_rate"], 0.0) or 0.0)
    w   = float(_safe_get(vstats, ["width"], 0.0) or 0.0)
    h   = float(_safe_get(vstats, ["height"], 0.0) or 0.0)
    fps = float(_safe_get(vstats, ["src_fps"], _safe_get(meta, ["fps"], 0.0)) or 0.0)
    denom = max(w * h * fps, 1.0)
    return br / denom

def _compression_from_bpp(bpp: float) -> Tuple[str, float]:
    # Soglie stimate per mobile H.264/HEVC consumer
    if bpp < 0.03:
        return ("heavy", 0.85)
    if bpp < 0.08:
        return ("medium", 0.55)
    return ("low", 0.20)

def _natural_capture_gate(vsum: Dict[str, float]) -> bool:
    motion = float(vsum.get("motion_avg", 0.0))
    flow   = float(vsum.get("optflow_mag_avg", 0.0))
    dup    = float(vsum.get("dup_avg", 1.0))
    return (motion > 12.0) and (flow > 1.0) and (dup < 0.80)

def _video_base_score(vstats: Dict[str, Any]) -> float:
    # Se il video analyzer ha prodotto una timeline_ai per-secondo, usane la media
    tl_ai = vstats.get("timeline_ai")
    if isinstance(tl_ai, list) and tl_ai:
        vals = [float(x.get("ai_score", 0.5)) for x in tl_ai if isinstance(x, dict)]
        if vals:
            return float(sum(vals) / len(vals))
    # Fallback (conservativo) da summary
    s = vstats.get("summary", {}) or {}
    motion = float(s.get("motion_avg", 0.0))
    flow   = float(s.get("optflow_mag_avg", 0.0))
    dup    = float(s.get("dup_avg", 0.0))
    # Più motion/flow => più "reale" ⇒ score AI più basso
    base = 0.5 \
         + 0.15 * (0.5 - min(motion / 25.0, 1.0)) \
         + 0.10 * (0.5 - min(flow   / 3.0,  1.0)) \
         + 0.10 * max(min(dup - 0.5, 0.5), -0.5)
    return _clamp(base, 0.0, 1.0)

def _audio_base_score(astats: Dict[str, Any]) -> float:
    tl = astats.get("timeline") or []
    if tl:
        vals = [float(x.get("ai_score", 0.5)) for x in tl if isinstance(x, dict)]
        if vals:
            return float(sum(vals) / len(vals))
    return float(astats.get("scores", {}).get("audio_mean", 0.5))

def fuse(vstats: Dict[str, Any], astats: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    meta   = payload.get("meta") or {}
    forensic = payload.get("forensic") or {}
    c2pa = bool(_safe_get(forensic, ["c2pa", "present"], False))

    v_base = _video_base_score(vstats)
    a_base = _audio_base_score(astats)

    # Pesi più equilibrati
    w_v, w_a, w_h = 0.45, 0.45, 0.10

    # Hints “soft”: nessun boost pro-AI; solo pull verso 0.5 e impatto confidenza
    bpp_val = _bpp(meta, vstats)
    comp_label, _ = _compression_from_bpp(bpp_val)

    pull = 0.0       # quanto stringere verso 0.5 (incertezza)
    conf_pen = 0.0   # penalità di confidenza
    if comp_label == "heavy":
        pull += 0.12
        conf_pen += 0.20
    elif comp_label == "medium":
        pull += 0.06
        conf_pen += 0.10

    # Natural capture safety: limita picchi AI su riprese mobili reali
    if _natural_capture_gate(vstats.get("summary", {})):
        v_cap = 0.70
    else:
        v_cap = 1.00

    # Gating audio su condizioni di silenzio/energia bassa
    flags_audio = astats.get("flags_audio") or []
    if ("mostly_silent" in flags_audio) or ("very_low_energy" in flags_audio):
        w_a *= 0.65

    # Fusione base (hints neutri = 0.5)
    base_ai = (w_v * min(v_base, v_cap)) + (w_a * a_base) + (w_h * 0.5)

    # Pull verso 0.5 in funzione della compressione/ambiguità
    base_ai = 0.5 + (base_ai - 0.5) * (1.0 - pull)
    base_ai = _clamp(base_ai, 0.0, 1.0)

    # Effetto C2PA: solo direzione pro-reale (e lieve calo confidenza)
    if c2pa:
        base_ai = 0.5 + (base_ai - 0.5) * 0.85
        conf_pen += 0.05

    # Timeline aggregata neutra (compatibilità UI)
    timeline_binned = []
    peaks = []
    duration = int(round(float(_safe_get(vstats, ["duration"], 0.0) or 0.0)))
    duration = max(1, min(duration, 180))
    for t in range(duration):
        timeline_binned.append({"start": t, "end": t + 1, "ai_score": float(base_ai)})
        peaks.append({"t": t, "ai_score": float(base_ai)})

    # Label e confidenza
    ai_score = float(base_ai)
    if ai_score >= THRESH_AI:
        label = "ai"
    elif ai_score <= THRESH_REAL:
        label = "real"
    else:
        label = "uncertain"

    conf = 0.70
    if label == "ai":
        conf += 0.10
    elif label == "real":
        conf += 0.08
    conf = max(0.10, conf - conf_pen)
    confidence = int(round(_clamp(conf, 0.10, 0.99) * 100))

    reasons: List[str] = []
    if comp_label in ("medium", "heavy"):
        reasons.append(f"Compressione {comp_label}")
    if _natural_capture_gate(vstats.get("summary", {})):
        reasons.append("Dinamica da ripresa reale (safety-cap)")
    if c2pa:
        reasons.append("C2PA/Manifest presente")

    result = {
        "label": label,
        "ai_score": float(round(ai_score, 6)),
        "confidence": int(confidence),
        "reason": "; ".join(reasons) if reasons else "",
    }

    hints = {
        "bpp": float(round(bpp_val, 5)),
        "compression": comp_label,
        "video_has_signal": bool(vstats),
        "flow_used": float(round(float(_safe_get(vstats, ["summary", "optflow_mag_avg"], 0.0) or 0.0), 4)),
        "motion_used": float(round(float(_safe_get(vstats, ["summary", "motion_avg"], 0.0) or 0.0), 3)),
        "w": int(_safe_get(vstats, ["width"], 0) or _safe_get(meta, ["width"], 0) or 0),
        "h": int(_safe_get(vstats, ["height"], 0) or _safe_get(meta, ["height"], 0) or 0),
        "fps": float(_safe_get(meta, ["fps"], 0.0) or 0.0),
        "br": int(_safe_get(meta, ["bit_rate"], 0) or 0),
    }
    if c2pa:
        hints["c2pa_present"] = "Manifest/Content Credentials rilevati"

    return {
        "result": result,
        "timeline_binned": timeline_binned,
        "peaks": peaks,
        "hints": hints,
    }
