def _get(meta: dict, *path, default=None):
    cur = meta or {}
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default

def compute_hints(video_stats: dict = None, audio_stats: dict = None, meta: dict = None) -> dict:
    hints = {}
    v = video_stats or {}
    a = audio_stats or {}
    m = meta or {}

    # BPP
    br  = float(_get(m, "bit_rate", default=0.0) or 0.0)
    w   = float(_get(v, "width", default=0.0) or 0.0)
    h   = float(_get(v, "height", default=0.0) or 0.0)
    fps = float(_get(v, "src_fps", default=0.0) or _get(m, "fps", default=0.0) or 0.0)
    denom = (w*h*fps) if (w>0 and h>0 and fps>0) else 1.0
    bpp = br/denom
    hints["bpp"] = round(bpp, 5)

    if bpp < 0.03:
        hints["compression"] = "heavy"
    elif bpp < 0.08:
        hints["compression"] = "medium"
    else:
        hints["compression"] = "low"

    # Video summary
    vsum = v.get("summary", {}) or {}
    motion_avg = float(vsum.get("motion_avg", 0.0))
    edge_avg   = float(vsum.get("edge_var_avg", 0.0))
    dup_avg    = float(vsum.get("dup_avg", 0.0))
    block_avg  = float(vsum.get("blockiness_avg", 0.0))
    band_avg   = float(vsum.get("banding_avg", 0.0))
    flow_avg   = float(vsum.get("optflow_mag_avg", 0.0))

    if motion_avg < 0.15:
        hints["low_motion"] = True
    if edge_avg < 20.0:
        hints["low_texture"] = True
    if dup_avg > 0.7 and motion_avg < 0.5:
        hints["many_duplicates"] = True
    if block_avg > 0.35 or band_avg > 0.35:
        hints["heavy_compression_blocks"] = True
    if (flow_avg < 0.05 and motion_avg > 0.3) or (flow_avg > 1.5 and motion_avg < 0.2):
        hints["flow_motion_mismatch"] = True

    c2pa_present = bool(_get(m, "forensic", "c2pa", "present", default=False))
    if c2pa_present:
        hints["c2pa_present"] = True

    if w and h:
        ar = w / max(h, 1.0)
        if ar < 0.9 or ar > 2.2:
            hints["unusual_aspect_ratio"] = True

    return hints
