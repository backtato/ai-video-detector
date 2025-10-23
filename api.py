import os
import io
import json
import tempfile
import traceback
from typing import Optional, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

# --- Config da ENV (backward compat) ---
APP_NAME = os.getenv("APP_NAME", "ai-video-detector")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))  # 50 MB
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
DEFAULT_WINDOW_SEC = int(os.getenv("WINDOW_SEC", "3"))  # finestra segmenti
PREFILTERS_DEFAULT = os.getenv("PREFILTERS_DEFAULT", "0") == "1"

# --- Moduli esistenti (riuso) ---
# Import "morbidi": se la tua repo ha already questi moduli, li usiamo
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
        # fallback neutro
        return 0.5

try:
    from detectors.audio import score_audio  # type: ignore
except Exception:
    def score_audio(_path: str) -> float:
        return 0.45

# --- Nuovi helper additive (non invasivi) ---
from app.social import detect_source_profile  # nuovo file
from app.filters.pre import build_prefilter_fn  # nuovo file
from app.utils.segments import make_segments, analyze_segments  # nuovo file

# --- Calibrazione / combinazione (riuso se presenti) ---
try:
    from calibration import calibrate, combine_scores  # type: ignore
except Exception:
    def calibrate(scores: Dict[str, float]) -> Tuple[float, float]:
        # fallback conservativo
        parts = scores.copy()
        ai = 0.34 * parts.get("metadata", 0.5) + 0.33 * parts.get("frame_artifacts", 0.5) + 0.33 * parts.get("audio", 0.45)
        conf = 0.05  # prudente su social
        return ai, conf

    def combine_scores(scores: Dict[str, float]) -> float:
        return calibrate(scores)[0]

app = FastAPI(title=APP_NAME)

# --- CORS ---
allow_origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
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
    # Fallback minimo
    return {"programs": [], "stream_groups": [], "streams": [], "format": {}}

def _analyze_core(path: str,
                  apply_prefilters: bool,
                  social_mode: Optional[bool],
                  window_sec: int) -> Dict[str, Any]:
    info = _ffprobe_safe(path)

    # Rileva profilo sorgente
    source_profile = detect_source_profile(info, force_social=social_mode)

    # Pre-filter opzionale (solo frame_artifacts lo userà)
    prefilter_fn = build_prefilter_fn(enabled=apply_prefilters)

    # Punteggi "globali" (clip intero) -> riuso dei tuoi detectors
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

    # Analisi a finestre (diagnostica, non cambia l'AI score globale)
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

    # Combina punteggi (tua calibrazione se disponibile)
    ai_score, confidence = calibrate(parts)

    details = {
        "parts": parts,
        "ffprobe": info,
        "adaptive_weights": {
            "metadata": 0.34, "frame_artifacts": 0.33, "audio": 0.33
        },
        "source_profile": source_profile,
        "prefilters_applied": bool(prefilter_fn is not None),
        "segments": segments,  # elenco di finestre con mini-punteggi (diagnostica)
    }

    # Etichetta di contesto per la UI/UX (non blocca API)
    if source_profile == "social":
        details["note"] = "Modalità Social attiva: ricompressione forte (AV1/H.264, bitrate basso) → risultati prudenti."

    return {
        "ai_score": round(float(ai_score), 4),
        "confidence": float(confidence),
        "details": details,
    }

# --- Endpoints ---

@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")

@app.get("/predict")
@app.get("/predict-get")
def predict_get(url: str):
    # per retro-compatibilità
    raise HTTPException(status_code=405, detail="Use POST /predict with form-data (url or file).")

@app.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(default=None),
    url: Optional[str] = Form(default=None),
    cookies_b64: Optional[str] = Form(default=None),  # placeholder compat
    social_mode: Optional[bool] = Form(default=None),
    apply_prefilters: Optional[bool] = Form(default=None),
    window_sec: Optional[int] = Form(default=None),
):
    """
    Accetta: file **oppure** url.
    Parametri opzionali:
      - social_mode: forza Social Mode (True/False). Se None -> auto.
      - apply_prefilters: abilita pre-filtri anti-social (default da ENV).
      - window_sec: dimensione finestre segmenti (default 3s).
    """
    if not file and not url:
        raise HTTPException(status_code=400, detail="Provide 'file' or 'url'.")

    # Normalizza flags
    if apply_prefilters is None:
        apply_prefilters = PREFILTERS_DEFAULT
    if window_sec is None or window_sec <= 0:
        window_sec = DEFAULT_WINDOW_SEC

    # Risoluzione input -> salvataggio temporaneo
    tmp_path = None
    try:
        if file:
            if file.size and int(file.size) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"File too large (> {MAX_UPLOAD_BYTES} bytes).")
            raw = await file.read()
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"File too large (> {MAX_UPLOAD_BYTES} bytes).")
            fd, tmp_path = tempfile.mkstemp(prefix="aivd_", suffix=".bin")
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
        else:
            # URL: usa il tuo risolutore già presente a monte, oppure semplice fetch
            # Per sicurezza manteniamo approccio minimal, così non rompiamo Render.
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

# Alias WP (retro-compatibilità)
@app.post("/analyze")
async def analyze_alias(
    file: Optional[UploadFile] = File(default=None),
    url: Optional[str] = Form(default=None),
    cookies_b64: Optional[str] = Form(default=None),
):
    return await predict(file=file, url=url, cookies_b64=cookies_b64)

@app.post("/analyze-url")
async def analyze_url(
    url: str = Form(...),
    cookies_b64: Optional[str] = Form(default=None),
):
    return await predict(file=None, url=url, cookies_b64=cookies_b64)
