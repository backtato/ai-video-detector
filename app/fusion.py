# app/analyzers/fusion.py
from typing import Dict, Any, List, Tuple
import math

# Soglie conservative allineate alla tua UI 1.1.3
THRESH_REAL = float(0.35)
THRESH_AI   = float(0.72)

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))

def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def _norm(x: float, lo: float, hi: float) -> float:
    """Normalizza un valore in [lo,hi] nello spazio [0,1], clamp inclusi."""
    if hi <= lo:
        return 0.0
    return _clamp((float(x) - lo) / (hi - lo), 0.0, 1.0)

def _video_quality_hints(video: Dict[str, Any]) -> Tuple[float, Dict[str,str]]:
    """
    Deriva un 'compression_index' in [0,1] da blockiness/banding medi.
    0.0 = qualità ottima; 1.0 = fortemente compresso.
    """
    block_avg = _safe_get(video, ["summary","blockiness_avg"], 0.0) or 0.0
    band_avg  = _safe_get(video, ["summary","banding_avg"],   0.0) or 0.0

    # Queste soglie empiriche sono pensate per i range che vedo nei tuoi log
    # blockiness ~ 0.02..0.05 ; banding ~ 0.34..0.41
    block_n = _norm(block_avg, 0.02, 0.05)
    band_n  = _norm(band_avg,  0.34, 0.42)

    compression_index = _clamp(0.6 * block_n + 0.4 * band_n)
    notes = {}
    if compression_index >= 0.6:
        notes["heavy_compression"] = "Forte compressione (blockiness/banding)."
    elif compression_index >= 0.35:
        notes["moderate_compression"] = "Compressione moderata."
    return compression_index, notes

def _align_bins(n: int, vals: List[float]) -> List[float]:
    """Pad/tronca una lista per avere esattamente n elementi (replica ultimo)."""
    if n <= 0:
        return []
    if not vals:
        return [0.5] * n
    if len(vals) == n:
        return vals
    if len(vals) > n:
        return vals[:n]
    last = vals[-1]
    return vals + [last] * (n - len(vals))

def _bin_video(video: Dict[str, Any], n_bins: int) -> List[float]:
    """
    Produce un ai_score 'video-only' per bin, usando motion/dup/edge come proxy.
    È una proxy euristica debole: rimane vicina a 0.5 e si sposta poco.
    """
    timeline = video.get("timeline") or []
    if not timeline:
        return [0.5] * n_bins

    # Estrai feature per-secondo se disponibili (nostra timeline è già per secondo)
    mot = []
    dup = []
    edge = []
    for b in timeline:
        mot.append(float(b.get("motion", 0.0)))
        dup.append(float(b.get("dup", 0.0)))
        edge.append(float(b.get("edge_var", 0.0)))

    # Normalizzazioni robuste sui quantili/medie disponibili
    if mot:
        m_min, m_max = min(mot), max(mot)
        m_norm = [ _norm(x, m_min, m_max if m_max>m_min else (m_min+1.0)) for x in mot ]
    else:
        m_norm = []

    if dup:
        d_min, d_max = min(dup), max(dup)
        d_norm = [ _norm(x, d_min, d_max if d_max>d_min else (d_min+1.0)) for x in dup ]
    else:
        d_norm = []

    if edge:
        e_min, e_max = min(edge), max(edge)
        e_norm = [ _norm(x, e_min, e_max if e_max>e_min else (e_min+1.0)) for x in edge ]
    else:
        e_norm = []

    L = min(len(m_norm) or 10**9, len(d_norm) or 10**9, len(e_norm) or 10**9)
    if L == 10**9:  # nessuna feature utile
        per = [0.5] * len(timeline)
    else:
        m_norm, d_norm, e_norm = m_norm[:L], d_norm[:L], e_norm[:L]
        per = []
        for i in range(L):
            # euristica: più dup (frame ripetuti) + edge poveri = più 'sospetto' (ma poco)
            # motion alto in compresso introduce aliasing → sposta leggermente.
            v = 0.5 \
                + 0.04 * (d_norm[i] - 0.5) \
                + 0.03 * (0.5 - e_norm[i]) \
                + 0.03 * (m_norm[i] - 0.5)
            per.append(_clamp(v))

    per = _align_bins(n_bins, per)
    return per

def _confidence_from_spread(series: List[float], has_audio: bool, compression_index: float) -> float:
    """
    Confidenza conservativa: cresce se i valori si discostano in modo consistente da 0.5
    e se abbiamo audio; scende se compressione è alta.
    """
    if not series:
        return 0.25
    mean = sum(series)/len(series)
    spread = sum(abs(x - 0.5) for x in series)/len(series)  # deviazione media da 0.5
    # base: 30% di spread + bonus 10% se abbiamo audio, penalità compressione fino a -15%
    base = 0.30 * (spread / 0.5)  # normalizza: 0.0→0%; 0.5→30%
    bonus_audio = 0.10 if has_audio else 0.0
    penalty_comp = -0.15 * compression_index
    conf = 0.30 + base + bonus_audio + penalty_comp
    return _clamp(conf, 0.10, 0.99)

def _label_from_score(score: float) -> str:
    if score <= THRESH_REAL:
        return "real"
    if score >= THRESH_AI:
        return "ai"
    return "uncertain"

def fuse(video: Dict[str, Any], audio: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combina segnali video+audio e produce:
    {
      "label": str, "ai_score": float, "confidence": float, "reason": str,
      "timeline_binned": [{"start":s,"end":e,"ai_score":f}, ...],
      "peaks": []
    }
    Chiavi e struttura compatibili con la tua UI (v1.1.3).
    """
    duration = float(_safe_get(meta, ["duration"], _safe_get(video, ["duration"], 0.0)) or 0.0)
    n_bins = max(1, int(math.ceil(duration)))

    # 1) Stima compressione video → hint qualitativo
    compression_index, comp_notes = _video_quality_hints(video)

    # 2) Timeline audio AI (già in [0,1] per-secondo dal nuovo audio analyzer)
    a_tl = [float(b.get("ai_score", 0.5)) for b in (audio.get("timeline") or [])]
    a_tl = _align_bins(n_bins, a_tl)
    a_mean = float(sum(a_tl)/len(a_tl)) if a_tl else 0.5
    has_audio = bool(audio.get("timeline"))

    # 3) Timeline video euristica (debole): rimane vicina a 0.5
    v_tl = _bin_video(video, n_bins)
    v_mean = float(sum(v_tl)/len(v_tl)) if v_tl else 0.5

    # 4) Fusione conservativa per-secondo:
    #    - audio dominante (0.65) perché porta informazione più “umana”
    #    - video debole (0.25) per non introdurre bias su compressione
    #    - penalità/offset piccolo dalla compressione (±0.10 centrato su 0.5)
    fused: List[float] = []
    for i in range(n_bins):
        f = 0.65 * a_tl[i] + 0.25 * v_tl[i] + 0.10 * (0.5 + 0.5*(a_tl[i] - 0.5))  # piccolo boost verso la direzione audio
        # leggera trazione verso 0.5 se compressione alta (evita falsi positivi)
        pull = 0.05 * compression_index
        f = (1.0 - pull) * f + pull * 0.5
        fused.append(_clamp(f))

    # 5) Score globale = media pesata (audio>video) + lieve regolarizzazione
    ai_score = float(sum(fused)/len(fused)) if fused else 0.5
    ai_score = _clamp(0.85 * ai_score + 0.15 * (0.5 + 0.5*(a_mean - 0.5)))

    # 6) Confidenza
    confidence = _confidence_from_spread(fused, has_audio=has_audio, compression_index=compression_index)

    # 7) Label conservativa, come da tua UI
    label = _label_from_score(ai_score)

    # 8) Reason sintetico
    reasons: List[str] = []
    if "heavy_compression" in comp_notes:
        reasons.append("Indizi di bassa qualità: heavy_compression")
    elif "moderate_compression" in comp_notes:
        reasons.append("Qualità video moderatamente compressa")
    # hint audio
    tts_like = float(_safe_get(audio, ["scores","tts_like"], 0.0) or 0.0)
    hnr_proxy = float(_safe_get(audio, ["scores","hnr_proxy"], 0.0) or 0.0)
    if tts_like >= 0.35:
        reasons.append("Profilo audio piatto/omogeneo (tts-like)")
    if hnr_proxy <= 0.55:
        reasons.append("Basso rapporto armonico-rumore")
    if not reasons:
        reasons.append("Segnali misti o neutri")

    # 9) timeline_binned per UI (stesso formato già in uso)
    timeline_binned = [{"start": float(i), "end": float(i+1), "ai_score": float(fused[i])} for i in range(n_bins)]

    return {
        "label": label,
        "ai_score": ai_score,
        "confidence": confidence,
        "reason": "; ".join(reasons),
        "timeline_binned": timeline_binned,
        "peaks": []  # opzionale: puoi popolarlo se in futuro vuoi mostrare picchi > soglia
    }
