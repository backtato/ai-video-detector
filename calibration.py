from typing import Dict

def squash(x: float) -> float:
    """
    Simple logistic-like squashing to map raw [0,1] → softer [0,1].
    Keeps monotonicity, reduces overconfidence.
    """
    # y = 1/(1+exp(-k*(x-0.5))) scaled back to [0,1]; approximate without math.exp for speed.
    # Use a polynomial proxy: 3x^2 - 2x^3 (smoothstep), gives S-shaped curve.
    return max(0.0, min(1.0, 3 * (x ** 2) - 2 * (x ** 3)))

def combine_scores(parts: Dict[str, float], weights: Dict[str, float], calibrate: bool = True) -> float:
    total_w = sum(weights.values()) or 1.0
    score = 0.0
    for k, v in parts.items():
        w = weights.get(k, 0.0) / total_w
        score += w * v
    return squash(score) if calibrate else max(0.0, min(1.0, score))
