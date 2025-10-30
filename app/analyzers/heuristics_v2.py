def compute_hints(meta: dict, path: str) -> dict:
    width = meta.get("width") or 0
    height = meta.get("height") or 0
    fps = meta.get("fps") or 0.0
    bit_rate = meta.get("bit_rate") or 0

    pixels_per_sec = (width*height*fps) if width and height and fps else 0.0
    bpp = float(bit_rate)/max(1.0, pixels_per_sec)
    if bpp <= 0.04: comp = "very_heavy"
    elif bpp <= 0.08: comp = "heavy"
    elif bpp <= 0.15: comp = "normal"
    else: comp = "light"

    return {
        "w": width, "h": height, "fps": fps, "br": bit_rate,
        "bpp": round(bpp,5), "compression": comp,
        "video_has_signal": (width*height) > 0 and fps > 0,
        "dup_avg": 0.0
    }