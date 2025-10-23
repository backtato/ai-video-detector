from typing import Optional, Callable

def build_prefilter_fn(enabled: bool = False):
    if not enabled:
        return None
    try:
        import cv2  # lazy import
        import numpy as np  # noqa: F401

        def prefilter(frame_bgr):
            f = cv2.fastNlMeansDenoisingColored(frame_bgr, None, 3, 3, 7, 21)
            try:
                f = cv2.detailEnhance(f, sigma_s=3, sigma_r=0.2)
            except Exception:
                pass
            return f

        return prefilter
    except Exception:
        # OpenCV assente/non avviabile → nessun prefilter
        return None
