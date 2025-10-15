import math

def logistic(x, a, b):
    return 1.0 / (1.0 + math.exp(-(a*x + b)))

def calibrate(raw, a, b):
    return logistic(raw, a, b)

def combine_scores(weighted):
    num=0.0; den=0.0
    for _, (score,w) in weighted.items():
        num += score*w; den += w
    return (num/den) if den>0 else 0.5
