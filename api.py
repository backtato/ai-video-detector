import os
import io
import json
import tempfile
import traceback
from typing import Optional, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi import params as fastapi_params  # <— per rilevare oggetti Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

APP_NAME = os.getenv("APP_NAME", "ai-video-detector")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))  # 50 MB
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGIN_REGEX = os.getenv("ALLOWED_ORIGIN_REGEX", "")  # opzionale
DEFAULT_WINDOW_SEC = int(os.getenv("WINDOW_SEC", "3"))
PREFILTERS_DEFAULT = os.getenv("PREFILTERS_DEFAULT", "0") == "1"

# --- Moduli esistenti (riuso, fail-safe) ---
try:
    from detectors.metadata import ffprobe, score_metadata  # type: ignore
except Exception:
    ffprobe = None
    def score_metadata(_path: str) -> float:
        return 0.5

try:
    from detectors.frame_artifacts import score_frame_artifacts  # type: ignore
except Exception:
    def score_frame_artifacts(_path: str, t_start: Optional[float] = None, t_end: Optional[float] = None,
                              prefilter_fn=None) -> float:
        return 0.5

try:
    from detectors.audio import score_audio  # type: ignore
except Exception:
    def score_audio(_path: str) -> float:
        return 0.45

# --- Nuovi helper (import lazy + fallback) ---
def _import_social():
    try:
        from app.social import detect_source_profile
        return detect_source_profile
    except Exception:
        def _fallback_detect_source_profile(_ffp: Dict[str, Any], force_social: Optional[bool] = None) -> str:
            return "social" if force_social else "clean"
        return _fallback_detect_source_profile

def _import_prefilter_builder():
    try:
        from app.filters.pre import build_prefilter_fn
        return build_prefilter_fn
    except Exception:
        def _noop_builder(enabled: bool = False):
            return None
        return _noop_builder

def _import_segments():
    try:
        from app.utils.segments import make_segments, analyze_segments
        return make_segments, analyze_segments
    except Exception:
        def _make_segments(_total: float, _window_sec: int = 3):
            return []
        def _analyze_segments(_path: str, _segments, _prefilter_fn=None):
            return []
        return _make_segments, _analyze_segments

# --- Calibrazione / combinazione (riuso se presenti) ---
try:
    from calibration import calibrate, combine_scores  # type: ignore
except Exception:
    def calibrate(scores: Dict[str, float]) -> Tuple[float, float]:
        ai = 0.34 * scores.get("metadata", 0.5) + 0.33 * scores.get("frame_artifacts", 0.5) + 0.33 * scores.get("audio", 0.45)
        conf = 0.05
        return ai, conf
    def combine_scores(scores: Dict[str, float]) -> float:
        return calibrate(scores)[0]

app = FastAPI(title=APP_NAME)

# --- CORS robusto e auto-safe ---
allow_origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()] or ["*"]
use_wildcard = (allow_origins == ["*"])
allow_origin_regex = ALLOWED_ORIGIN_REGEX or None
allow_credentials = False if use_wildcard else True

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if use_wildcard else allow_origins,
    allow_origin_regex=allow_origin_regex,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Utils ---
def _ffprobe_safe(path: str) -> Dict[str, Any]:
    if ffprobe:
        try:
            return ffprobe(path)
        except Exception:
            pass
    return {"programs": [], "stream_groups": [], "streams": [], "format": {}}

def _analyze_core(path: str,
                  apply_prefilters: bool,
                  social_mode: Optional[bool],
                  window_sec: int) -> Dict[str, Any]:
    detect_source_profile = _import_social()
    build_prefilter_fn = _import_prefilter_builder()
    make_segments, analyze_segments = _import_segments()

    info = _ffprobe_safe(path)
    source_profile = detect_source_profile(info, force_social=social_mode)

    # Prefiltro (se disponibile)
    prefilter_fn = build_prefilter_fn(enabled=apply_prefilters)

    # Punteggi globali
    try:
        s_meta = float(score_metadata(path))
    except Exception:
        s_meta = 0.5
    try:
        s_frame = float(score_frame_artifacts(path, prefilter_fn=prefilter_fn))
    except Exception:
        s_frame = 0.5
    try:
        s_audio = float(score_audio(path))
    except Exception:
        s_audio = 0.45

    parts = {"metadata": s_meta, "frame_artifacts": s_frame, "audio": s_audio}

    # Segmenti (diagnostica)
    segments = []
    try:
        t_total = 0.0
        if isinstance(info.get("format", {}).get("duration"), str):
            try:
                t_total = float(info["format"]["duration"])
            except Exception:
                t_total = 0.0
        segments_idx = make_segments(t_total, window_sec=window_sec)
        segments = analyze_segments(path, segments_idx, prefilter_fn=prefilter_fn)
    except Exception:
        segments = []

    ai_score, confidence = calibrate(parts)

    details = {
        "parts": parts,
        "ffprobe": info,
        "adaptive_weights": {"metadata": 0.34, "frame_artifacts": 0.33, "audio": 0.33},
        "source_profile": source_profile,
        "prefilters_applied": bool(prefilter_fn is not None),
        "segments": segments,
    }
    if source_profile == "social":
        details["note"] = "Modalità Social: ricompressione forte (AV1/H.264) → risultati prudenti."

    return {
        "ai_score": round(float(ai_score), 4),
        "confidence": float(confidence),
        "details": details,
    }

# --- Health & diagnostica leggera ---
@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")

@app.get("/")
def root():
    return PlainTextResponse("ok")

@app.get("/env-cors")
def env_cors():
    return {"ALLOWED_ORIGINS_raw": os.getenv("ALLOWED_ORIGINS", "*"),
            "parsed": allow_origins, "credentials": allow_credentials}

# --- API principali ---
@app.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(default=None),
    url: Optional[str] = Form(default=None),
    cookies_b64: Optional[str] = Form(default=None),
    social_mode: Optional[bool] = Form(default=None),
    apply_prefilters: Optional[bool] = Form(default=None),
    window_sec: Optional[int] = Form(default=None),
):
    # Se predict è chiamata direttamente (non via HTTP), i parametri possono essere oggetti Form → normalizza a None
    if isinstance(social_mode, fastapi_params.Form):       social_mode = None
    if isinstance(apply_prefilters, fastapi_params.Form):  apply_prefilters = None
    if isinstance(window_sec, fastapi_params.Form):        window_sec = None

    if not file and not url:
        raise HTTPException(status_code=400, detail="Provide 'file' or 'url'.")

    if apply_prefilters is None:
        apply_prefilters = PREFILTERS_DEFAULT
    if window_sec is None or int(window_sec) <= 0:
        window_sec = DEFAULT_WINDOW_SEC

    tmp_path = None
    try:
        if file:
            if getattr(file, "size", None) and int(file.size) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"File too large (> {MAX_UPLOAD_BYTES} bytes).")
            raw = await file.read()
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"File too large (> {MAX_UPLOAD_BYTES} bytes).")
            fd, tmp_path = tempfile.mkstemp(prefix="aivd_", suffix=".bin")
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
        else:
            import requests
            r = requests.get(url, timeout=20, headers={"User-Agent": "AI-Video/1.0"})
            r.raise_for_status()
            content = r.content
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Downloaded media too large.")
            fd, tmp_path = tempfile.mkstemp(prefix="aivd_", suffix=".bin")
            with os.fdopen(fd, "wb") as f:
                f.write(content)

        result = _analyze_core(
            path=tmp_path,
            apply_prefilters=bool(apply_prefilters),
            social_mode=social_mode,
            window_sec=int(window_sec),
        )
        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# Alias retro-compatibili — passano esplicitamente i parametri opzionali a None
@app.post("/analyze")
async def analyze_alias(
    file: Optional[UploadFile] = File(default=None),
    url: Optional[str] = Form(default=None),
    cookies_b64: Optional[str] = Form(default=None),
):
    return await predict(
        file=file,
        url=url,
        cookies_b64=cookies_b64,
        social_mode=None,
        apply_prefilters=None,
        window_sec=None,
    )

@app.post("/analyze-url")
async def analyze_url(
    url: str = Form(...),
    cookies_b64: Optional[str] = Form(default=None),
):
    return await predict(
        file=None,
        url=url,
        cookies_b64=cookies_b64,
        social_mode=None,
        apply_prefilters=None,
        window_sec=None,
    )