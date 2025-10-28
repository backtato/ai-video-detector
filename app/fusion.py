# app/fusion.py
import math
from typing import Dict, Any, List
import numpy as np

THRESH_REAL = 0.35
THRESH_AI   = 0.72

# pesi conservativi
W_VIDEO = 0.55
W_AUDIO = 0.35
W_HINTS = 0.10  # l'AI-score non usa hint in modo diretto; gli hint pesano su reason/confidence

def _clamp(x: float, lo: float=0.0, hi: float=1.0) -> float:
    return max(lo, min(hi, x))

def _safe(d: Dict, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None or not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default

def _quality_from_video(vstats: dict, meta: dict) -> float:
    """Stima 0..1 della qualità (1 = ottima). Penalizza duplicati, blockiness, banding."""
    s = vstats.get('summary', {})
    blockiness = float(s.get('blockiness_avg', 0.0))
    banding    = float(s.get('banding_avg', 0.0))
    dup_avg    = float(s.get('dup_avg', 0.0))

    def norm(v, lo, hi):
        try:
            return _clamp((v - lo) / (hi - lo))
        except Exception:
            return 0.0

    # taratura conservativa per WhatsApp/screen-rec
    q_penalty  = 0.5*norm(blockiness, 0.02, 0.06) \
               + 0.3*norm(banding,    0.30, 0.42) \
               + 0.4*norm(dup_avg,    0.88, 0.98)
    q = max(0.0, 1.0 - q_penalty)
    return q

def _video_ai_per_second(vstats: dict) -> List[float]:
    """Heuristica leggera per ricavare un 'sospetto AI' per secondo dal video.
    Baseline 0.5. Aumenta leggermente con pattern inconsueti; riduce con motion vera."""
    tl = vstats.get('timeline', [])
    if not tl:
        return []
    scores = []
    for sec in tl:
        motion      = float(sec.get('motion', 0.0))
        dup         = float(sec.get('dup', 0.0))
        blockiness  = float(sec.get('blockiness', 0.0))
        banding     = float(sec.get('banding', 0.0))
        s = 0.5
        s +=  0.04 if banding > 0.40 else 0.0
        s +=  0.03 if blockiness > 0.04 else 0.0
        s +=  0.03 if dup > 0.95 else 0.0
        s -=  0.05 if motion > 15.0 else 0.0
        scores.append(_clamp(s, 0.0, 1.0))
    return scores

def _audio_ai_per_second(astats: dict) -> List[float]:
    tl = _safe(astats, 'timeline', default=[])
    if not tl:
        return []
    return [float(sec.get('ai_score', 0.5)) for sec in tl]

def _combine_per_second(v: List[float], a: List[float]) -> List[float]:
    n = max(len(v), len(a))
    if n == 0:
        return []
    out = []
    v_mean = (sum(v)/len(v)) if v else 0.5
    a_mean = (sum(a)/len(a)) if a else 0.5
    for i in range(n):
        vs  = v[i] if i < len(v) else v_mean
        as_ = a[i] if i < len(a) else a_mean
        s = W_VIDEO*vs + W_AUDIO*as_ + W_HINTS*0.5
        out.append(_clamp(s, 0.0, 1.0))
    return out

def _peaks(timeline: List[float], min_score: float = 0.55, min_len: int = 2) -> List[Dict[str, float]]:
    peaks = []
    start = None
    for i, val in enumerate(timeline):
        if val >= min_score:
            if start is None:
                start = i
        else:
            if start is not None and (i - start) >= min_len:
                peaks.append({'start': float(start), 'end': float(i), 'score': float(np.mean(timeline[start:i]))})
            start = None
    if start is not None and (len(timeline) - start) >= min_len:
        peaks.append({'start': float(start), 'end': float(len(timeline)), 'score': float(np.mean(timeline[start:]))})
    return peaks

def fuse(video_stats: dict, audio_stats: dict, hints: dict, meta: dict) -> Dict[str, Any]:
    # per-second fusion
    v_per = _video_ai_per_second(video_stats)
    a_per = _audio_ai_per_second(audio_stats)
    fused = _combine_per_second(v_per, a_per)

    # ai_score finale come media dei secondi analizzati
    ai_score = float(np.mean(fused)) if fused else 0.5

    # qualità e confidence
    q = _quality_from_video(video_stats, meta)
    margin = abs(ai_score - 0.5) / 0.5  # 0..1
    confidence = 0.35 + 0.65 * margin * q
    confidence = float(_clamp(confidence, 0.0, 1.0))

    # label conservativa + quality gate
    if ai_score <= THRESH_REAL:
        label = 'real'
    elif ai_score >= THRESH_AI and q >= 0.55:
        label = 'ai'
    else:
        label = 'uncertain'

    # peaks per UI (segmenti da rivedere)
    pk = _peaks(fused, min_score=0.55, min_len=2)

    # reason
    reasons: List[str] = []
    if q < 0.35:
        reasons.append('Qualità limitante (compressione/duplicati): valutazione prudente.')
    if hints:
        neg = [k for k in hints.keys() if 'heavy' in k or 'low_' in k or 'very_low' in k]
        pos = [k for k in hints.keys() if 'c2pa_present' in k or 'authentic' in k]
        if pos:
            reasons.append('Indizi positivi: ' + ', '.join(pos))
        if neg:
            reasons.append('Indizi di bassa qualità: ' + ', '.join(neg))
    if pk:
        segs = [f"{int(p['start'])}-{int(p['end'])}s" for p in pk[:3]]
        reasons.append('Segmenti borderline da rivedere: ' + ', '.join(segs))

    return {
        'result': {
            'label': label,
            'ai_score': float(ai_score),
            'confidence': float(confidence),
            'reason': ' '.join(reasons) if reasons else 'Valutazione conservativa.'
        },
        'timeline_binned': [
            {'start': float(i), 'end': float(i+1), 'ai_score': float(s)} for i, s in enumerate(fused)
        ],
        'peaks': pk
    }
