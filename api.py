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
# Config da ENV
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

# =======================
# Calibrazione / combinazione
# =======================
ENSEMBLE_WEIGHTS = {"metadata": 0.33, "frame_artifacts": 0.34, "audio": 0.33}
CALIBRATION = None

# Importa prima dalla root (nel repo i file sono in root), poi da app/
try:
    from calibration import calibrate as _calibrate  # type: ignore
    from calibration import combine_scores as _combine_scores  # type: ignore
except Exception:
    try:
        from app.calibration import calibrate as _calibrate  # type: ignore
        from app.calibration import combine_scores as _combine_scores  # type: ignore
    except Exception:
        def _calibrate(x, *args, **kwargs):
            return float(x), 0.5
        def _combine_scores(d, *args, **kwargs):
            vals = [float(v) for v in d.values() if v is not None]
            return ((sum(vals) / len(vals)) if vals else 0.5, d)

# Carica pesi/calibrazione opzionali da config se presenti
try:
    from config import ENSEMBLE_WEIGHTS as _EW  # type: ignore
    if _EW: ENSEMBLE_WEIGHTS = _EW
except Exception:
    try:
        from app.config import ENSEMBLE_WEIGHTS as _EW  # type: ignore
        if _EW: ENSEMBLE_WEIGHTS = _EW
    except Exception:
        pass

try:
    from config import CALIBRATION as _CAL  # type: ignore
    CALIBRATION = _CAL
except Exception:
    try:
        from app.config import CALIBRATION as _CAL  # type: ignore
        CALIBRATION = _CAL
    except Exception:
        pass

# =======================
# Detectors (fallback neutrali)
# =======================
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
# Helpers I/O
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
    Scarica via HTTP(S) grezzo quando l'URL è già un file .mp4/.webm/.mov/.mkv
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
    1) URL diretto (.mp4/.webm/.mov/.mkv) → HTTP fallback
    2) Altrimenti yt-dlp (client android/web); cookies_b64 opzionale (per admin)
    """
    lower = url.lower().split("?", 1)[0]
    if any(lower.endswith(ext) for ext in (".mp4", ".webm", ".mov", ".mkv")):
        return _http_fallback(url)

    if not yt_dlp:
        raise HTTPException(status_code=500, detail="yt-dlp non installato nel server")

    tmp_dir = tempfile.mkdtemp(prefix="aivideo_")
    outtmpl = os.path.join(tmp_dir, "download.%(ext)s")
    fmt = "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b"

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
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
    }

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
        if cookie_file:
            try: os.remove(cookie_file)
            except Exception: pass

# ---------- Normalizzazione score ----------
def _flatten_scores_dict(raw_mixed: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
  """
  Converte un dict con valori misti (float, tuple, dict, ecc.) in:
    - raw_float: solo float per combine_scores
    - parts: dettagli secondari (per JSON)
  """
  raw_float: Dict[str, float] = {}
  parts: Dict[str, Any] = {}
  for k, v in raw_mixed.items():
      score_val: float = 0.5
      extra: Any = {}
      try:
          if isinstance(v, (tuple, list)) and len(v) > 0:
              try: score_val = float(v[0])
              except Exception: score_val = 0.5
              if len(v) > 1: extra = {"extra": v[1:]}
          elif isinstance(v, dict) and ("score" in v):
              try: score_val = float(v.get("score", 0.5))
              except Exception: score_val = 0.5
              extra = v
          elif isinstance(v, (int, float)):
              score_val = float(v)
              extra = {}
          elif isinstance(v, str):
              try: score_val = float(v)
              except Exception: score_val = 0.5
              extra = {"raw": v}
          else:
              score_val = 0.5
              extra = {"raw": repr(v)}
      except Exception:
          score_val = 0.5
          extra = {"raw": "coercion_error"}
      if score_val < 0.0: score_val = 0.0
      if score_val > 1.0: score_val = 1.0
      raw_float[k] = score_val
      parts[k] = extra
  return raw_float, parts

# ---------- Adattività conservativa ----------
def _extract_quality(meta: Dict[str, Any]) -> Dict[str, float]:
    """Estrae indicatori rozzi di qualità dal ffprobe (se disponibili)."""
    width = height = fps = bitrate_kbps = 0.0
    try:
        for s in (meta.get("streams") or []):
            if s.get("codec_type") == "video":
                width = float(s.get("width", width) or width)
                height = float(s.get("height", height) or height)
                fr = s.get("avg_frame_rate") or "0/0"
                if isinstance(fr, str) and "/" in fr:
                    a, b = fr.split("/", 1)
                    fb = float(b or 1.0)
                    fps = (float(a) / fb) if fb != 0 else fps
        fmt = meta.get("format") or {}
        br = fmt.get("bit_rate") or fmt.get("bitrate") or 0
        try: bitrate_kbps = float(br) / 1000.0
        except Exception: bitrate_kbps = 0.0
    except Exception:
        pass
    dur = 0.0
    try: dur = float((meta.get("format") or {}).get("duration") or 0.0)
    except Exception: dur = 0.0
    return {"width": width, "height": height, "fps": fps, "bitrate_kbps": bitrate_kbps, "duration": dur}

def _quality_factor(q: Dict[str, float]) -> float:
    """
    Fattore 0..1 di qualità (molto conservativo).
    Valori bassi solo se bitrate < 900 kbps e/o risoluzione < 720p.
    """
    w, h = q["width"], q["height"]
    br = q["bitrate_kbps"]
    fps = q["fps"]

    # risoluzione: <720p penalizza, >=1080p ok
    max_side = max(w, h)
    res_score = 0.0
    if max_side <= 0:
        res_score = 0.0
    elif max_side < 720:
        res_score = max(0.0, (max_side - 360) / (720 - 360))  # 360p~0 → 720p~1
    elif max_side >= 1080:
        res_score = 1.0
    else:
        res_score = 0.75 + 0.25 * ((max_side - 720) / (1080 - 720))

    # bitrate: <900 kbps penalizza; >2500 ok
    br_score = 0.0 if br <= 0 else min(1.0, max(0.0, (br - 900) / (2500 - 900)))

    # fps: <20 penalizza, 30 ok
    fps_score = 0.0 if fps <= 0 else min(1.0, max(0.0, (fps - 20) / (30 - 20)))

    # combinazione conservativa
    qf = 0.5 * br_score + 0.35 * res_score + 0.15 * fps_score
    return max(0.0, min(1.0, qf))

def _adaptive_weights(base: Dict[str, float], parts: Dict[str, float], meta: Dict[str, Any]) -> Dict[str, float]:
    """
    Riduce il peso di 'frame_artifacts' SOLO se qualità è bassa (bitrate < 900 kbps o <720p).
    Riduzione massima 40%, altrimenti nessuna riduzione. Delta redistribuito su metadata/audio.
    """
    q = _extract_quality(meta)
    qf = _quality_factor(q)  # 0..1
    w = dict(base)

    low_quality = (q["bitrate_kbps"] > 0 and q["bitrate_kbps"] < 900) or (max(q["width"], q["height"]) > 0 and max(q["width"], q["height"]) < 720)
    if low_quality:
        # fattore tra 0.6 (qualità pessima) e 1.0 (qualità buona) → riduzione max 40%
        frame_factor = 0.6 + 0.4 * qf
    else:
        frame_factor = 1.0

    w_frame_old = w.get("frame_artifacts", 0.0)
    w_frame_new = w_frame_old * frame_factor
    delta = max(0.0, w_frame_old - w_frame_new)
    w["frame_artifacts"] = w_frame_new

    # Redistribuisci delta su metadata/audio proporzionalmente ai loro pesi
    pool = {k: w.get(k, 0.0) for k in ("metadata", "audio") if k in w}
    pool_sum = sum(pool.values())
    if delta > 0 and pool_sum > 0:
        for k in pool:
            w[k] = w.get(k, 0.0) + delta * (pool[k] / pool_sum)

    # Normalizza a somma 1
    tot = sum(w.values()) or 1.0
    for k in w:
        w[k] = w[k] / tot
    return w

def _agreement_factor(vals: Dict[str, float]) -> float:
    """
    Accordo tra segnali: se sono molto diversi, la confidenza scende.
    Ritorna 0.6..1.0 (conservativo).
    """
    try:
        xs = [float(v) for v in vals.values()]
        if not xs: return 0.7
        mu = sum(xs)/len(xs)
        var = sum((x-mu)**2 for x in xs)/len(xs)
        f = 1.0 - min(0.4, var*3.0)  # var più grande → abbassa max di 0.4
        return max(0.6, min(1.0, f))
    except Exception:
        return 0.7

# =======================
# Analisi
# =======================
def _analyze_video(path: str) -> Dict[str, Any]:
    """
    Esegue i detector, normalizza in float, combina con pesi ADATTIVI (conservativi) e calibra.
    Confidenza finale modulata da durata, qualità e accordo tra segnali.
    """
    try:
        meta = ffprobe(path)
    except Exception:
        meta = {}

    # punteggi grezzi
    try: m_val = score_metadata(meta)
    except Exception: m_val = 0.5
    try: f_val = score_frame_artifacts(path)
    except Exception: f_val = 0.5
    try: a_val = score_audio(path)
    except Exception: a_val = 0.5

    raw_mixed = { "metadata": m_val, "frame_artifacts": f_val, "audio": a_val }
    raw_float, extra_parts = _flatten_scores_dict(raw_mixed)

    # Pesi adattivi (conservativi)
    adaptive_w = _adaptive_weights(ENSEMBLE_WEIGHTS, raw_float, meta)

    # Combina (la tua combine_scores richiede SEMPRE i pesi)
    combined, comb_parts = _combine_scores(raw_float, adaptive_w)
    if not isinstance(comb_parts, dict):
        comb_parts = extra_parts

    # Calibra (supporta sia calibrate(score, params) che calibrate(score))
    try:
        if CALIBRATION is not None:
            calibrated, confidence = _calibrate(combined, CALIBRATION)
        else:
            calibrated, confidence = _calibrate(combined)
    except TypeError:
        calibrated, confidence = _calibrate(combined)

    # Confidenza robusta e conservativa
    q = _extract_quality(meta)
    dur = q["duration"] or 0.0
    # durata: <=10s → 0.7; 10–20s → 0.7..0.95; >20s → 1.0
    if dur <= 10:
        dur_factor = 0.7
    elif dur >= 20:
        dur_factor = 1.0
    else:
        dur_factor = 0.7 + (dur - 10) / 10 * 0.25  # 0.70→0.95

    qual_factor = 0.7 + 0.3 * _quality_factor(q)  # 0.7..1.0
    agree_factor = _agreement_factor(raw_float)    # 0.6..1.0

    confidence = float(confidence) * dur_factor * qual_factor * agree_factor
    confidence = max(0.05, min(1.0, confidence))

    return {
        "ai_score": round(float(calibrated), 4),
        "confidence": round(float(confidence), 4),
        "details": {
            "parts": comb_parts,              # punteggi per canale (post-normalizzazione)
            "ffprobe": meta,
            "adaptive_weights": adaptive_w,   # per trasparenza/debug
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