from typing import Dict, Tuple, Any

def _coerce_float(v: Any, default: float = 0.5) -> float:
    """Converte valori misti (tuple/dict/str/num) in float nello [0,1]."""
    try:
        # tuple/list: prendi il primo elemento
        if isinstance(v, (tuple, list)) and len(v) > 0:
            v = v[0]
        # dict: se ha 'score', prendi quello
        if isinstance(v, dict) and ("score" in v):
            v = v.get("score", default)
        # stringa numerica?
        if isinstance(v, str):
            try:
                v = float(v)
            except Exception:
                v = default
        # numerico già ok
        if isinstance(v, (int, float)):
            v = float(v)
        else:
            v = default
    except Exception:
        v = default

    # clamp 0..1
    if v < 0.0: v = 0.0
    if v > 1.0: v = 1.0
    return v

def combine_scores(raw: Dict[str, Any], weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    Combina gli score pesati. Si aspetta SEMPRE 'weights'.
    Ritorna:
      - combined: float 0..1
      - parts: dict con gli score normalizzati per chiave (dopo coerce)
    """
    if not isinstance(raw, dict):
        raise TypeError("raw deve essere un dict")
    if not isinstance(weights, dict):
        raise TypeError("weights deve essere un dict")

    total_w = 0.0
    s = 0.0
    parts: Dict[str, float] = {}

    for k, w in weights.items():
        try:
            w = float(w)
        except Exception:
            w = 0.0
        if w <= 0.0:
            continue

        v = _coerce_float(raw.get(k, 0.5), default=0.5)
        parts[k] = v
        total_w += w
        s += v * w

    if total_w <= 0.0:
        # Nessun peso valido: fallback media semplice dei raw coerced
        vals = [_coerce_float(v) for v in raw.values()]
        combined = sum(vals) / len(vals) if vals else 0.5
    else:
        combined = s / total_w

    # clamp finale
    if combined < 0.0: combined = 0.0
    if combined > 1.0: combined = 1.0

    return float(combined), parts

def calibrate(score: float, params: Dict[str, float] = None) -> Tuple[float, float]:
    """
    Calibrazione semplice + stima confidence.
    - Se params contiene 'scale' e 'bias', applichiamo y = clamp(score*scale + bias)
    - Confidence = max(0.05, |score-0.5|*2), in [0.05..1.0]
    """
    try:
        s = float(score)
    except Exception:
        s = 0.5

    scale = 1.0
    bias = 0.0
    if isinstance(params, dict):
        try: scale = float(params.get("scale", scale))
        except Exception: pass
        try: bias = float(params.get("bias", bias))
        except Exception: pass

    y = s * scale + bias
    if y < 0.0: y = 0.0
    if y > 1.0: y = 1.0

    conf = abs(y - 0.5) * 2.0
    if conf < 0.05: conf = 0.05
    if conf > 1.0: conf = 1.0

    return float(y), float(conf)