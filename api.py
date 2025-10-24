# api.py
import os
import io
import json
import math
import shutil
import tempfile
import traceback
import subprocess
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# --- opzionale: yt-dlp per link social pubblici ---
try:
    from yt_dlp import YoutubeDL  # type: ignore
    HAS_YTDLP = True
except Exception:
    HAS_YTDLP = False

# --- opzionale: OpenCV per fallback fps (non obbligatorio se c'è ffprobe) ---
try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

# =========================
# Config da ENV
# =========================
APP_NAME = os.getenv("APP_NAME", "ai-video-detector")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))  # 50MB
RESOLVER_MAX_BYTES = int(os.getenv("RESOLVER_MAX_BYTES", str(120 * 1024 * 1024)))
REQUEST_TIMEOUT_S = int(os.getenv("REQUEST_TIMEOUT_S", "120"))
USE_YTDLP = os.getenv("USE_YTDLP", "1") in ("1", "true", "True", "yes", "YES")

# CORS
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["*"]  # dev fallback

# UA per le richieste HTTP
RESOLVER_UA = os.getenv("RESOLVER_UA", "Mozilla/5.0 (AVD/1.0; +https://ai-video.org)")

# Opzioni yt-dlp da ENV
_raw = os.getenv("YTDLP_OPTS", "")
try:
    YTDLP_OPTS = json.loads(_raw) if _raw else {}
except Exception:
    YTDLP_OPTS = {}

DEFAULT_YTDLP_OPTS = {
    "noplaylist": True,
    "continuedl": False,
    "retries": 1,
    "quiet": True,
}

# Domini consentiti
ALLOW_DOMAINS = set([
    "youtube.com", "www.youtube.com", "youtu.be",
    "vimeo.com", "www.vimeo.com",
    "instagram.com", "www.instagram.com",
    "tiktok.com", "www.tiktok.com",
    "facebook.com", "www.facebook.com", "fb.watch",
    "x.com", "www.x.com", "twitter.com", "www.twitter.com",
])
ALLOW_EXTS = (".mp4", ".mov", ".m4v", ".webm", ".mpg", ".mpeg", ".avi")

# =========================
# App & CORS
# =========================
app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Utils
# =========================
def _http_headers():
    return {"User-Agent": RESOLVER_UA}

def _is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False

def _tmpfile(suffix: str = "") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path

def _raise_422(msg: str):
    raise HTTPException(status_code=422, detail=msg)

# ---------- ffprobe helpers ----------
def _ffprobe_json(path: str) -> dict:
    """Ritorna il JSON di ffprobe (format + streams)."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_format", "-show_streams",
            "-print_format", "json",
            path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return json.loads(out.decode("utf-8", "ignore"))
    except Exception:
        return {}

def _stream_fps(stream: dict) -> Optional[float]:
    # prefer avg_frame_rate
    afr = stream.get("avg_frame_rate") or ""
    try:
        if afr and afr != "0/0":
            num, den = afr.split("/")
            num, den = float(num), float(den)
            if den != 0:
                return num / den
    except Exception:
        pass
    # fallback r_frame_rate
    rfr = stream.get("r_frame_rate") or ""
    try:
        if rfr and rfr != "0/0":
            num, den = rfr.split("/")
            num, den = float(num), float(den)
            if den != 0:
                return num / den
    except Exception:
        pass
    return None

def _probe_meta_rich(path: str) -> dict:
    """Estrae metadati ricchi stile 'meta' precedente, più un blocco semplice per la UI attuale."""
    info = _ffprobe_json(path)
    fmt = info.get("format", {}) or {}
    streams = info.get("streams", []) or []

    vstreams = [s for s in streams if s.get("codec_type") == "video"]
    astreams = [s for s in streams if s.get("codec_type") == "audio"]
    v = vstreams[0] if vstreams else {}
    a = astreams[0] if astreams else {}

    # meta ricco
    width = int(v.get("width") or 0) or None
    height = int(v.get("height") or 0) or None
    fps = _stream_fps(v)
    try:
        duration = float(fmt.get("duration")) if fmt.get("duration") else None
    except Exception:
        duration = None
    try:
        bit_rate = int(fmt.get("bit_rate")) if fmt.get("bit_rate") else None
    except Exception:
        bit_rate = None
    vcodec = v.get("codec_name")
    acodec = a.get("codec_name")
    format_name = fmt.get("format_name")
    tags = (fmt.get("tags") or {}) | (v.get("tags") or {}) | (a.get("tags") or {})

    # forensic (stub c2pa + quicktime tags)
    apple_qt_tags = any(k.lower().startswith("com.apple.quicktime") for k in tags.keys())
    forensic = {
        "c2pa": {"present": False},  # TODO: integrazione libreria C2PA se/quando disponibile
        "apple_quicktime_tags": bool(apple_qt_tags),
        "flags": []
    }

    meta = {
        "width": width, "height": height, "fps": fps,
        "duration": duration, "bit_rate": bit_rate,
        "vcodec": vcodec, "acodec": acodec, "format_name": format_name,
        "source_url": None, "resolved_url": None
    }

    # anche la forma "video" minimale per la UI nuova
    video_min = {"width": width, "height": height, "fps": fps}

    return meta, forensic, video_min

def _build_timelines(duration: Optional[float]) -> tuple[list, list]:
    """Crea timeline/timeline_binned coerenti con la durata (stub score=0.50)."""
    if not duration or duration <= 0:
        return [], []
    # finestra mobile ogni ~0.5s (stub)
    step = 0.5
    win = 1.0
    t = 0.0
    timeline = []
    while t < duration + 0.5:
        timeline.append({
            "start": round(max(0.0, t - win/2), 6),
            "end":   round(min(duration + 0.5, t + win/2), 6),
            "ai_score": 0.50
        })
        t += step
    # binned per secondi interi
    bins = []
    sec = 0.0
    while sec < duration:
        end = min(duration, sec + 1.0)
        bins.append({"start": round(sec, 6), "end": round(end, 6), "ai_score": 0.50})
        sec += 1.0
    return timeline, bins

def _basic_fusion_scores(path: str) -> dict:
    """Placeholder di punteggi. Senza modello ML manteniamo valori conservativi."""
    # Potresti arricchire con euristiche: rolling shutter, motion blur, bitrate/fps mismatch, ecc.
    scores = {"frame": 0.50, "audio": 0.50}
    fusion = {"ai_score": 0.50, "label": "uncertain", "confidence": 0.50}
    return scores, fusion

def _probe_metadata_cv2(path: str) -> dict:
    """Solo fallback width/height/fps se ffprobe mancasse (non dovrebbe)."""
    meta = {"width": None, "height": None, "fps": None}
    if not HAS_CV2: return meta
    cap = cv2.VideoCapture(path)
    try:
        if cap.isOpened():
            meta["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
            meta["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            meta["fps"] = float(fps) if fps else None
    finally:
        cap.release()
    return meta

def _rich_analyze(path: str, source_url: Optional[str] = None, resolved_url: Optional[str] = None) -> dict:
    """Restituisce sia lo schema 'ricco' (compat pre-aggiornamento) sia lo schema 'nuovo' (result/video)."""
    meta, forensic, video_min = _probe_meta_rich(path)
    if source_url is not None:
        meta["source_url"] = source_url
    if resolved_url is not None:
        meta["resolved_url"] = resolved_url

    # punteggi placeholder (no ML)
    scores, fusion = _basic_fusion_scores(path)

    # timeline stub coerenti con la durata
    timeline, timeline_binned = _build_timelines(meta.get("duration"))

    # blocco "nuovo" per la UI corrente
    result_new = {
        "label": fusion["label"],            # "uncertain"
        "ai_score": fusion["ai_score"],      # 0.50
        "confidence": fusion["confidence"],  # 0.50
    }
    video_new = video_min

    # output ricco + nuovo
    out = {
        "ok": True,
        "meta": meta,
        "forensic": forensic,
        "scores": scores,
        "fusion": fusion,
        "timeline": timeline,
        "timeline_binned": timeline_binned,
        "peaks": [],

        # compat con la UI nuova
        "result": result_new,
        "video": video_new
    }
    return out

# ---------- Upload / Download ----------
def _save_upload_to_tmp(upload: UploadFile) -> str:
    size = 0
    tmp_path = _tmpfile(suffix=os.path.splitext(upload.filename or "")[1] or ".bin")
    with open(tmp_path, "wb") as out:
        while True:
            chunk = upload.file.read(1 << 20)  # 1MB
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                try: os.remove(tmp_path)
                except Exception: pass
                _raise_422(f"File troppo grande (> {MAX_UPLOAD_BYTES} bytes)")
            out.write(chunk)
    try: upload.file.close()
    except Exception: pass
    return tmp_path

def _download_direct(url: str) -> Tuple[str, Optional[str], Optional[str]]:
    with requests.Session() as s:
        try:
            h = s.head(url, headers=_http_headers(), allow_redirects=True, timeout=REQUEST_TIMEOUT_S)
        except Exception as e:
            _raise_422(f"Errore rete: {e}")

        cl = int(h.headers.get("Content-Length", "0") or "0")
        ctype = (h.headers.get("Content-Type") or "").lower()
        if cl and cl > RESOLVER_MAX_BYTES:
            _raise_422("Il file remoto è troppo grande. Usa 'Registra 15s' o 'Carica file'.")
        if "text/html" in ctype:
            _raise_422("Not a direct video (text/html). Usa 'Carica file' o 'Registra 15s'.")

        try:
            r = s.get(url, headers=_http_headers(), stream=True, timeout=REQUEST_TIMEOUT_S)
            r.raise_for_status()
        except Exception as e:
            _raise_422(f"Errore download: {e}")

        ext = os.path.splitext(urlparse(r.url).path)[1].lower()
        if not ext:
            if "mp4" in ctype: ext = ".mp4"
            elif "webm" in ctype: ext = ".webm"
            elif "quicktime" in ctype or "mov" in ctype: ext = ".mov"
            else: ext = ".bin"

        tmp_path = _tmpfile(suffix=ext)
        size = 0
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk: break
                size += len(chunk)
                if size > RESOLVER_MAX_BYTES:
                    f.close()
                    try: os.remove(tmp_path)
                    except Exception: pass
                    _raise_422("Dimensione oltre il limite. Usa 'Registra 15s' o 'Carica file'.")
                f.write(chunk)
    # restituisci anche l’URL risolto finale
    return tmp_path, ctype or None, r.url

def _ytdlp_resolve_and_download(url: str) -> Tuple[str, Optional[str]]:
    if not HAS_YTDLP:
        _raise_422("yt-dlp non disponibile nel server.")
    opts = {**DEFAULT_YTDLP_OPTS, **YTDLP_OPTS}
    tmpdir = tempfile.mkdtemp()
    outtmpl = os.path.join(tmpdir, "%(title).80s.%(ext)s")
    opts.update({
        "outtmpl": outtmpl,
        "restrictfilenames": True,
        "nopart": True,
        "ignoreerrors": False,
        "user_agent": RESOLVER_UA,
        "merge_output_format": "mp4",
    })
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if not info:
                _raise_422("Impossibile estrarre il video (yt-dlp).")
            if "requested_downloads" in info and info["requested_downloads"]:
                fpath = info["requested_downloads"][0]["filepath"]
            else:
                title = info.get("title", "video")
                ext = info.get("ext", "mp4")
                fpath = os.path.join(tmpdir, f"{title}.{ext}")
            if not os.path.exists(fpath):
                cand = [os.path.join(tmpdir, p) for p in os.listdir(tmpdir)]
                cand = [p for p in cand if os.path.isfile(p)]
                if not cand: _raise_422("Download non riuscito (yt-dlp).")
                fpath = sorted(cand, key=lambda p: -os.path.getsize(p))[0]
            if os.path.getsize(fpath) > RESOLVER_MAX_BYTES:
                try: shutil.rmtree(tmpdir)
                except Exception: pass
                _raise_422("File troppo grande dal provider. Usa 'Registra 15s' o 'Carica file'.")
            suffix = os.path.splitext(fpath)[1] or ".mp4"
            final_path = _tmpfile(suffix=suffix)
            shutil.move(fpath, final_path)
            try: shutil.rmtree(tmpdir)
            except Exception: pass
            return final_path, info.get("webpage_url") or None
    except Exception as e:
        msg = str(e)
        if "login" in msg.lower() and "require" in msg.lower():
            _raise_422("DownloadError yt-dlp: login required / contenuto non pubblico")
        if "rate" in msg.lower() and "limit" in msg.lower():
            _raise_422("DownloadError yt-dlp: rate-limit reached dal provider")
        _raise_422(f"DownloadError yt-dlp: {msg}")

# =========================
# Endpoints
# =========================
@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")

@app.post("/analyze")
def analyze(file: UploadFile = File(...)):
    if not file:
        _raise_422("Nessun file fornito.")
    try:
        tmp_path = _save_upload_to_tmp(file)
        out = _rich_analyze(tmp_path)
        try: os.remove(tmp_path)
        except Exception: pass
        return JSONResponse(out)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Errore interno: {e}")

@app.post("/analyze-url")
async def analyze_url(request: Request):
    """
    Accetta:
    - JSON: { "url": "https://..." }
    - form: url=<...>
    - query: ?url=...
    """
    url_in: Optional[str] = None
    # JSON
    try:
        data = await request.json()
        if isinstance(data, dict) and "url" in data:
            url_in = str(data.get("url", "")).strip()
    except Exception:
        pass
    # form
    if not url_in:
        try:
            form = await request.form()
            if "url" in form:
                url_in = str(form.get("url", "")).strip()
        except Exception:
            pass
    # query
    if not url_in:
        url_in = str(request.query_params.get("url", "")).strip()

    if not url_in:
        _raise_422("Campo 'url' mancante.")
    if not _is_url(url_in):
        _raise_422("URL non valido.")

    parsed = urlparse(url_in)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    is_direct = any(path.endswith(ext) for ext in ALLOW_EXTS)

    if (host not in ALLOW_DOMAINS) and (not is_direct):
        _raise_422("Dominio non supportato. Usa 'Carica file' o 'Registra 15s'.")

    try:
        resolved = None
        if is_direct:
            tmp_path, _ctype, resolved = _download_direct(url_in)
        else:
            if not USE_YTDLP:
                _raise_422("Richiede estrazione dal provider. Attiva USE_YTDLP o usa 'Carica file'/'Registra 15s'.")
            tmp_path, resolved = _ytdlp_resolve_and_download(url_in)

        out = _rich_analyze(tmp_path, source_url=url_in, resolved_url=resolved)
        try: os.remove(tmp_path)
        except Exception: pass
        return JSONResponse(out)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Errore interno: {e}")

@app.post("/predict")
def predict(file: UploadFile = File(None), url: Optional[str] = Form(None)):
    if file is not None:
        return analyze(file=file)
    if url:
        _raise_422("Usa /analyze-url (JSON {url} o form) o /analyze con file.")
    _raise_422("Fornire 'file' oppure 'url'.")