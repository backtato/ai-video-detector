def build_hints(meta: dict, video: dict, audio: dict, base_hints: dict) -> dict:
    """
    Arricchisce gli hints esistenti con segnali utili alla fusione.
    - handheld_camera_likely se iOS/Apple + flow/motion presenti + bpp ok
    - video_has_signal è in base_hints
    """
    hints = dict(base_hints or {})
    bpp = hints.get("bpp") or 0.0
    comp = hints.get("compression")
    flow = hints.get("flow_used") or 0.0
    motion = hints.get("motion_used") or 0.0

    dev = (meta or {}).get("device") or {}
    if (dev.get("vendor") == "Apple" or dev.get("os") == "iOS") and hints.get("video_has_signal", False):
        if bpp >= 0.08 and (comp in (None, "normal", "low")) and (flow > 1.0 or motion > 22.0):
            hints["handheld_camera_likely"] = True
        else:
            hints["handheld_camera_likely"] = False
    else:
        hints["handheld_camera_likely"] = False

    v_sum = (video or {}).get("summary") or {}
    hints["flow_used"] = float(v_sum.get("optflow_mag_avg", hints.get("flow_used") or 0.0))
    hints["motion_used"] = float(v_sum.get("motion_avg", hints.get("motion_used") or 0.0))
    return hints
