import os
import json
import tempfile
import subprocess
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import numpy as np
import soundfile as sf
import cv2
try:
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

from app.analyzers import audio as audio_an
from app.analyzers import video as video_an
from app.analyzers import forensic as forensic_an
from app.analyzers import meta as meta_an
from app.analyzers import fusion as fusion_an
from app.analyzers import heuristics_v2 as heur_an

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
REQUEST_TIMEOUT_S = int(os.getenv("REQUEST_TIMEOUT_S", "240"))

app = FastAPI()

allow_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins if allow_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _run_ffprobe(path: str) -> Dict[str, Any]:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=30)
        return json.loads(out.decode("utf-8", errors="ignore"))
    except Exception:
        return {}

def _probe_basic(meta_json: Dict[str, Any]) -> Dict[str, Any]:
    w = h = fps = dur = br = 0
    vcodec = acodec = fmt = None
    try:
        for st in meta_json.get("streams", []):
            if st.get("codec_type") == "video":
                w = int(st.get("width") or 0)
                h = int(st.get("height") or 0)
                r = st.get("r_frame_rate") or st.get("avg_frame_rate") or "0/1"
                try:
                    a, b = r.split("/")
                    fps = float(a) / max(1.0, float(b))
                except Exception:
                    fps = float(st.get("avg_frame_rate") or 0) or 0.0
                vcodec = st.get("codec_name")
                dur = float(st.get("duration") or meta_json.get("format", {}).get("duration") or 0.0)
            elif st.get("codec_type") == "audio":
                acodec = st.get("codec_name")
        fmt = (meta_json.get("format") or {}).get("format_name")
        br = int((meta_json.get("format") or {}).get("bit_rate") or 0)
    except Exception:
        pass
    return dict(width=w, height=h, fps=fps, duration=dur, bit_rate=br, vcodec=vcodec, acodec=acodec, format_name=fmt)

def _try_cv_meta(path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {}
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = float(frame_count / fps) if fps > 0 else 0.0
    cap.release()
    return {"width": w, "height": h, "fps": fps, "duration": dur}

def _extract_wav_16k(path: str):
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        cmd = ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000", "-f", "wav", tmp.name]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        wav, sr = sf.read(tmp.name)
        os.unlink(tmp.name)
        if wav.ndim > 1:
            wav = wav[:, 0]
        return wav.astype(np.float32), sr
    except Exception:
        return None, 0

def _calc_bpp(bit_rate: int, w: int, h: int, fps: float) -> float:
    if not bit_rate or not w or not h or not fps:
        return 0.0
    px_per_s = w * h * fps
    return float(bit_rate) / float(px_per_s)

def _compression_from_bpp(bpp: float) -> str:
    if bpp <= 0.06:
        return "heavy"
    if bpp <= 0.11:
        return "normal"
    return "low"

def _analyze_file(path: str) -> Dict[str, Any]:
    meta_json = _run_ffprobe(path)
    basic = _probe_basic(meta_json)

    # fallback meta con OpenCV
    if basic["width"] == 0 or basic["height"] == 0 or basic["fps"] == 0:
        cvb = _try_cv_meta(path)
        for k in ("width", "height", "fps", "duration"):
            if (not basic.get(k)) and cvb.get(k):
                basic[k] = cvb[k]

    # AUDIO
    wav, sr = _extract_wav_16k(path)
    duration_audio = float(len(wav) / sr) if (wav is not None and sr > 0) else 0.0
    audio = audio_an.analyze(wav, sr, duration_audio) if wav is not None else {
        "scores": {"audio_mean": 0.5}, "flags_audio": ["low_voice_presence"], "timeline": []
    }

    # durata robusta
    if not basic["duration"] or basic["duration"] == 0:
        if duration_audio > 0:
            basic["duration"] = duration_audio

    # backfill bit_rate da filesize/durata
    try:
        if (not basic["bit_rate"] or basic["bit_rate"] == 0) and basic["duration"] and basic["duration"] > 0:
            fsize_bits = os.path.getsize(path) * 8.0
            basic["bit_rate"] = int(fsize_bits / basic["duration"])
    except Exception:
        pass

    # sanity-cap fps anomali
    if basic["fps"] and basic["fps"] > 120.0:
        basic["fps"] = 60.0

    # VIDEO
    video_has_signal = bool(basic["width"] and basic["height"] and basic["fps"])
    if video_has_signal:
        video = video_an.analyze(path, basic["fps"], basic["duration"])
    else:
        video = {"timeline": [], "summary": {}}
    if not video.get("timeline"):
        video_has_signal = False

    # HINTS di base
    bpp = _calc_bpp(basic.get("bit_rate") or 0, basic.get("width") or 0, basic.get("height") or 0, basic.get("fps") or 0.0)
    hints = {
        "bpp": bpp,
        "compression": _compression_from_bpp(bpp) if bpp else "heavy",
        "video_has_signal": video_has_signal,
        "w": basic.get("width") or 0,
        "h": basic.get("height") or 0,
        "fps": basic.get("fps") or 0,
        "br": basic.get("bit_rate") or 0
    }

    # clamp timeline a durata
    dur = float(basic.get("duration") or duration_audio or 0.0)
    def _clamp_tl(tl):
        out = []
        for x in tl or []:
            s = float(x.get("start", 0))
            e = float(x.get("end", s + 1))
            if s >= dur:
                continue
            e = min(e, dur)
            out.append({"start": s, "end": e, "ai_score": float(x.get("ai_score", 0.5))})
        return out

    v_summary = video.get("summary") or {}
    video_out = {
        "timeline": _clamp_tl(video.get("timeline") or []),
        "summary": v_summary,
        "timeline_ai": [{"start": t["start"], "end": t["end"], "ai_score": t["ai_score"]} for t in _clamp_tl(video.get("timeline") or [])]
    }
    audio_out = {
        "scores": audio.get("scores") or {},
        "flags_audio": audio.get("flags_audio") or [],
        "timeline": _clamp_tl(audio.get("timeline") or [])
    }

    # DEVICE: unwrap se il detector ritorna {"device": {...}}
    dev = meta_an.detect_device(path) if hasattr(meta_an, "detect_device") else {"vendor": None, "model": None, "os": None}
    if isinstance(dev, dict) and "device" in dev and isinstance(dev["device"], dict):
        dev = dev["device"]

    meta_out = {
        "width": basic["width"], "height": basic["height"], "fps": basic["fps"], "duration": basic["duration"],
        "bit_rate": basic["bit_rate"], "vcodec": basic["vcodec"], "acodec": basic["acodec"], "format_name": basic["format_name"],
        "source_url": None, "resolved_url": None,
        "forensic": {"c2pa": {"present": False}},
        "device": dev
    }

    hints["flow_used"] = float(v_summary.get("optflow_mag_avg") or 0.0)
    hints["motion_used"] = float(v_summary.get("motion_avg") or 0.0)

    hints = heur_an.build_hints(meta_out, video_out, audio_out, hints)
    fused = fusion_an.fuse(meta_out, hints, video_out, audio_out)

    out = {
        "ok": True,
        "meta": meta_out,
        "forensic": {"c2pa": {"present": meta_out.get("forensic", {}).get("c2pa", {}).get("present", False)}},
        "video": video_out,
        "audio": audio_out,
        "hints": hints,
        "result": fused["result"],
        "timeline_binned": fused["timeline_binned"],
        "peaks": fused["peaks"]
    }
    return out

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=415, detail={"error": "File vuoto o non ricevuto"})
    data = await file.read()
    if not data:
        raise HTTPException(status_code=415, detail={"error": "File vuoto"})
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail={"error": "File troppo grande"})

    # PATCH: usa estensione .mp4 per facilitare i demuxer
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        with open(tmp.name, "wb") as f:
            f.write(data)
        out = _analyze_file(tmp.name)
        return JSONResponse(out)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

@app.post("/analyze-url")
async def analyze_url(url: str = Form(None), link: str = Form(None), q: str = Form(None)):
    u = url or link or q
    if not u:
        raise HTTPException(status_code=422, detail="URL mancante")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()
    try:
        cmd = ["yt-dlp", "-f", "mp4", "-o", tmp.name, u]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        out = _analyze_file(tmp.name)
        out["meta"]["source_url"] = u
        out["meta"]["resolved_url"] = u
        return JSONResponse(out)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=422, detail=f"DownloadError yt-dlp: {e}")
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

@app.post("/predict")
async def predict(file: UploadFile = File(None), url: str = Form(None)):
    if file is not None:
        return await analyze(file)
    if url:
        return await analyze_url(url=url)
    raise HTTPException(status_code=422, detail="Fornire file o url")

# FACOLTATIVO: diagnostica CORS / preflight universale
@app.get("/healthz")
def healthz():
    return JSONResponse({"ok": True, "ffprobe": True, "exiftool": True, "version": "1.2.2", "author": "Backtato"})

# === Health/CORS minimal endpoints inserted ===

@app.get("/", response_class=JSONResponse)
def root():
    return {"ok": True, "service": "ai-video-detector", "version": VERSION}


from functools import lru_cache
from time import perf_counter
import shutil as _shutil

@lru_cache(maxsize=1)
def _ready_probe():
    t0 = perf_counter()
    ffprobe = _shutil.which("ffprobe") is not None
    exiftool = _shutil.which("exiftool") is not None
    return {"ffprobe_found": ffprobe, "exiftool_found": exiftool, "elapsed_ms": int((perf_counter()-t0)*1000)}

@app.get("/readyz", response_class=JSONResponse)
def readyz():
    return {"ok": True, **_ready_probe()}


@app.post("/cors-test", response_class=JSONResponse)
async def cors_test():
    return {"ok": True, "message": "CORS OK"}


@app.options("/{path:path}")
async def options_preflight(path: str):
    from fastapi.responses import Response
    return Response(status_code=204)
    # === Lightweight health/CORS endpoints + lazy import + handlers ===
from fastapi.responses import JSONResponse
import importlib

# /healthz super-leggero (per Render)
@app.get("/healthz", response_class=JSONResponse)
def healthz():
    return {"ok": True}

# /readyz (diagnostico, NON usato da Render)
from functools import lru_cache
from time import perf_counter
import shutil as _shutil

@lru_cache(maxsize=1)
def _ready_probe():
    t0 = perf_counter()
    return {
        "ffprobe_found": _shutil.which("ffprobe") is not None,
        "exiftool_found": _shutil.which("exiftool") is not None,
        "elapsed_ms": int((perf_counter()-t0)*1000),
    }

@app.get("/readyz", response_class=JSONResponse)
def readyz():
    return {"ok": True, **_ready_probe()}

# Banner root (facoltativo, utile)
@app.get("/", response_class=JSONResponse)
def root():
    return {"ok": True, "service": "ai-video-detector", "version": globals().get("VERSION", "unknown")}

# Preflight universale (CORS)
@app.options("/{path:path}")
async def options_preflight(path: str):
    from fastapi.responses import Response
    return Response(status_code=204)

# CORS quick test dal frontend
@app.post("/cors-test", response_class=JSONResponse)
async def cors_test():
    return {"ok": True, "message": "CORS OK"}

# Lazy import per librerie pesanti (evita cold-start lenti)
_IMPORT_CACHE = {}
def _lazy_import(name: str):
    mod = _IMPORT_CACHE.get(name)
    if mod is None:
        mod = importlib.import_module(name)
        _IMPORT_CACHE[name] = mod
    return mod

# Helper opzionali (usali nel codice dove richiesto: _np(), _sf(), _cv2())
def _np():
    return _lazy_import("numpy")

def _sf():
    return _lazy_import("soundfile")

def _cv2():
    return _lazy_import("cv2")

# Handlers JSON puliti (422/HTTPException)
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Invalid request", "errors": str(exc)}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail if isinstance(exc.detail, (str, dict)) else str(exc)}
    )
    