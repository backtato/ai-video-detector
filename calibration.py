import math
from typing import Dict

def logistic(x: float, a: float, b: float) -> float:
    # 1 / (1 + exp(-(a*x + b)))
    return 1.0 / (1.0 + math.exp(-(a * x + b)))

def calibrate(raw_score: float, a: float, b: float) -> float:
    return logistic(raw_score, a, b)

def combine_scores(weighted: Dict[str, float]) -> float:
    # weighted average of detector scores in [0,1]
    num = 0.0
    den = 0.0
    for k, v in weighted.items():
        num += v[0] * v[1]  # score * weight
        den += v[1]
    return (num / den) if den > 0 else 0.5
