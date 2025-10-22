import os
import shutil
import tempfile
import traceback
import base64
from typing import Optional, Tuple, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# =======================
# Config da ENV (source of truth in questo file)
# =======================
MAX_UPLOAD_BYTES     = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))   # 50 MB
RESOLVER_MAX_BYTES   = int(os.getenv("RESOLVER_MAX_BYTES", str(120 * 1024 * 1024)))
REQUEST_TIMEOUT_S    = int(os.getenv("REQUEST_TIMEOUT_S", "120"))
ALLOWED_ORIGINS_ENV  = os.getenv("ALLOWED_ORIGINS", "*")  # "*" oppure lista separata da virgole
YTDLP_COOKIES_B64ENV = os.getenv("YTDLP_COOKIES_B64", "")

# =======================
# Dipendenze opzionali
# =======================
try:
    import yt_dlp  # type: ignore
except Exception:
    yt_dlp = None  # Il server può girare anche senza yt-dlp (solo URL diretti)

# --- Calibrazione / combinazione (import robusti + fallback) ---
# Prova prima app.config, poi config alla root
ENSEMBLE_WEIGHTS = {"metadata": 0.33, "frame_artifacts": 0.34, "audio": 0.33}
CALIBRATION = None
try:
    from app.calibration import calibrate as _calibrate  # type: ignore
    from app.calibration import combine_scores as _combine_scores  # type: ignore
except Exception:
    try:
        from calibration import calibrate as _calibrate  # type: ignore
        from calibration import combine_scores as _combine_scores  # type: ignore
    except Exception:
        def _calibrate(x, *args, **kwargs):
            # ritorna (score_calibrato, confidenza)
            return float(x), 0.5
        def _combine_scores(d, *args, **kwargs):
            vals = [float(v) for v in d.values() if v is not None]
            return ((sum(vals) / len(vals)) if vals else 0.5, d)

# Prova a caricare i pesi e la calibrazione, prima da app.config poi da config
try:
    from app.config import ENSEMBLE_WEIGHTS as _EW  # type: ignore
    ENSEMBLE_WEIGHTS = _EW or ENSEMBLE_WEIGHTS
except Exception:
    try:
        from config import ENSEMBLE_WEIGHTS as _EW  # type: ignore
        ENSEMBLE_WEIGHTS = _EW or ENSEMBLE_WEIGHTS
    except Exception:
        pass
try:
    from app.config import CALIBRATION as _CAL  # type: ignore
    CALIBRATION = _CAL
except Exception:
    try:
        from config import CALIBRATION as _CAL  # type: ignore
        CALIBRATION = _CAL
    except Exception:
        pass

# --- Detectors (fallback neutrali se non disponibili) ---
try:
    from app.detectors.metadata import ffprobe, score_metadata  # type: ignore
except Exception:
    def ffprobe(path: str) -> Dict[str, Any]: return {}
    def score_metadata(meta: Dict[str, Any]) -> float: return 0.5

try:
    from app.detectors.frame_artifacts import score_frame_artifacts  # type: ignore
except Exception:
    def score_frame_artifacts(path: str) -> float: return 0.5

try:
    from app.detectors.audio import score_audio  # type: ignore
except Exception:
    def score_audio(path: str) -> float: return 0.5

# =======================
# FastAPI app + CORS
# =======================
app = FastAPI(title="AI Video Detector")

# CORS robusto: "*" => NO credenziali; lista => credenziali abilitate
if ALLOWED_ORIGINS_ENV.strip() == "*":
    allow_origins = ["*"]
    allow_credentials = False
else:
    allow_origins = [o.strip() for o in ALLOWED_ORIGINS_ENV.split(",") if o.strip()]
    allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# =======================
# Helpers
# =======================
def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "")[-1] or ".bin"
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    total = 0
    with os.fdopen(fd, "wb") as out:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                out.close()
                try: os.remove(temp_path)
                except Exception: pass
                raise HTTPException(status_code=413, detail=f"File too large (> {MAX_UPLOAD_BYTES} bytes)")
            out.write(chunk)
    return temp_path

def _http_fallback(url: str) -> str:
    """
    Scarica via HTTP(S) grezzo quando l'URL punta già a un file .mp4/.webm/.mov/.mkv
    """
    import requests
    tmpdir = tempfile.mkdtemp(prefix="aivideo_http_")
    path = os.path.join(tmpdir, "download.bin")
    try:
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT_S) as r:
            r.raise_for_status()
            size = 0
            with open(path, "wb") as out:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk: 
                        continue
                    size += len(chunk)
                    if size > RESOLVER_MAX_BYTES:
                        raise HTTPException(status_code=413, detail="Il video scaricato supera il limite server (fallback)")
                    out.write(chunk)
        if os.path.getsize(path) == 0:
            raise HTTPException(status_code=422, detail="File vuoto dopo download HTTP")
        return path
    except:
        try: shutil.rmtree(tmpdir, ignore_errors=True)
        except: pass
        raise

def _download_url_to_temp(url: str, cookies_b64: Optional[str] = None) -> str:
    """
    1) Se URL è diretto a file (.mp4/.webm/.mov/.mkv) → usa HTTP fallback
    2) Altrimenti prova yt-dlp con client 'android/web' (best effort)
       - se passato cookies_b64 (Base64 Netscape cookies.txt) → usalo
       - altrimenti se YTDLP_COOKIES_B64ENV è impostato → usalo
    """
    lower = url.lower().split("?", 1)[0]
    if any(lower.endswith(ext) for ext in (".mp4", ".webm", ".mov", ".mkv")):
        return _http_fallback(url)

    if not yt_dlp:
        raise HTTPException(status_code=500, detail="yt-dlp non installato nel server")

    tmp_dir = tempfile.mkdtemp(prefix="aivideo_")
    outtmpl = os.path.join(tmp_dir, "download.%(ext)s")
    fmt = "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b"  # robusto

    ydl_opts = {
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 2,
        "socket_timeout": 15,
        "user_agent": "Mozilla/5.0 (Android 10; Mobile) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari",
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
            }
        },
    }

    # Gestione cookie opzionale (per admin): per-request > env > nessuno
    cookie_file = None
    cookie_b64 = (cookies_b64 or "").strip() or (YTDLP_COOKIES_B64ENV.strip() if YTDLP_COOKIES_B64ENV else "")
    if cookie_b64:
        try:
            raw = base64.b64decode(cookie_b64)
            fd, cookie_file = tempfile.mkstemp(prefix="cookies_", text=True)
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            ydl_opts["cookiefile"] = cookie_file
        except Exception:
            # se i cookie sono malformati, ignora e prova senza
            cookie_file = None

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if not info:
                raise HTTPException(status_code=422, detail="Estrazione yt-dlp nulla (no info)")
            if info.get("requested_downloads"):
                filepath = info["requested_downloads"][0].get("filepath")
            else:
                ext = info.get("ext") or "mp4"
                filepath = outtmpl.replace("%(ext)s", ext)

            if not filepath or not os.path.exists(filepath):
                raise HTTPException(status_code=422, detail="File non creato da yt-dlp")

            size = os.path.getsize(filepath)
            if size == 0:
                raise HTTPException(status_code=422, detail="File vuoto dopo download")
            if size > RESOLVER_MAX_BYTES:
                raise HTTPException(status_code=413, detail="Il video scaricato supera il limite server")

            return filepath

    except yt_dlp.utils.DownloadError as e:
        msg = str(e) or ""
        hint = ""
        if ("Sign in to confirm you're not a bot" in msg) or ("Sign in to confirm you’re not a bot" in msg):
            hint = " (YouTube richiede cookie: puoi passare cookies_b64 per questa richiesta, oppure configurare YTDLP_COOKIES_B64 a livello server)"
        raise HTTPException(status_code=422, detail=f"DownloadError yt-dlp: {msg}{hint}")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=f"Risoluzione URL fallita: {str(e) or 'errore generico'}")
    finally:
        # pulizia cookies temporanei
        if cookie_file:
            try: os.remove(cookie_file)
            except Exception: pass

def _analyze_video(path: str) -> Dict[str, Any]:
    """
    Esegue i tre detector, combina, calibra e ritorna JSON.
    Compatibile sia con combine_scores(raw, weights) che con combine_scores(raw).
    Idem per calibrate(score, CALIBRATION) o calibrate(score).
    """
    # ffprobe + punteggi
    try:
        meta = ffprobe(path)
    except Exception:
        meta = {}

    try:
        s_meta = score_metadata(meta)
    except Exception:
        s_meta = 0.5

    try:
        s_frame = score_frame_artifacts(path)
    except Exception:
        s_frame = 0.5

    try:
        s_audio = score_audio(path)
    except Exception:
        s_audio = 0.5

    raw = {
        "metadata": s_meta,
        "frame_artifacts": s_frame,
        "audio": s_audio,
    }

    # combine_scores: prova con pesi; se la tua versione non li vuole, riprova senza
    try:
        combined, parts = _combine_scores(raw, ENSEMBLE_WEIGHTS)
    except TypeError:
        combined, parts = _combine_scores(raw)

    # calibrate: prova con parametri; se non servono, fallback
    try:
        if CALIBRATION is not None:
            calibrated, confidence = _calibrate(combined, CALIBRATION)
        else:
            calibrated, confidence = _calibrate(combined)
    except TypeError:
        calibrated, confidence = _calibrate(combined)

    return {
        "ai_score": round(float(calibrated), 4),
        "confidence": round(float(confidence), 4),
        "details": {
            "parts": parts,
            "ffprobe": meta,
            # opzionale: mostrare i pesi usati ai fini di trasparenza/debug
            # "config": {"weights": ENSEMBLE_WEIGHTS, "calibration": bool(CALIBRATION)}
        }
    }

# =======================
# Endpoints
# =======================
@app.get("/", response_class=HTMLResponse)
def index():
    return """<html><body>
    <h1>AI Video Detector</h1>
    <ul>
      <li>POST /predict (form: url OR file)</li>
      <li>POST /analyze (alias WP) (form: url OR file)</li>
      <li>POST /analyze-url (alias WP) (form: url)</li>
      <li>GET /predict-get?url=...</li>
      <li>GET /healthz</li>
    </ul>
    </body></html>"""

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

@app.get("/predict-get")
@app.get("/predict")
def predict_get(url: Optional[str] = None, cookies_b64: Optional[str] = None):
    if not url:
        raise HTTPException(status_code=400, detail="Parametro 'url' mancante")
    path = _download_url_to_temp(url, cookies_b64=cookies_b64)
    try:
        result = _analyze_video(path)
        return JSONResponse(result)
    finally:
        try: os.remove(path)
        except Exception: pass

@app.post("/predict")
async def predict(
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    cookies_b64: Optional[str] = Form(None),
):
    if not url and not file:
        raise HTTPException(status_code=400, detail="Fornisci 'url' oppure 'file'")
    temp_path = None
    try:
        if url:
            temp_path = _download_url_to_temp(url, cookies_b64=cookies_b64)
        else:
            if not file:
                raise HTTPException(status_code=400, detail="File mancante")
            temp_path = _save_upload_to_temp(file)
        result = _analyze_video(temp_path)
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Errore interno: {str(e) or 'analisi fallita'}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try: os.remove(temp_path)
            except Exception: pass

# Alias per plugin WordPress
@app.post("/analyze")
async def analyze_legacy(
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    cookies_b64: Optional[str] = Form(None),
):
    return await predict(url=url, file=file, cookies_b64=cookies_b64)

@app.post("/analyze-url")
async def analyze_url_legacy(
    url: Optional[str] = Form(None),
    cookies_b64: Optional[str] = Form(None),
):
    if not url:
        raise HTTPException(status_code=400, detail="Parametro 'url' mancante")
    return await predict(url=url, file=None, cookies_b64=cookies_b64)
