# api.py
import os
import io
import json
import shutil
import tempfile
import traceback
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# --- opzionale: yt-dlp per link social pubblici ---
try:
    from yt_dlp import YoutubeDL  # type: ignore
    HAS_YTDLP = True
except Exception:
    HAS_YTDLP = False

# --- opzionale: OpenCV per metadata video ---
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
    # fallback permissivo per sviluppo – in produzione imposta ALLOWED_ORIGINS
    ALLOWED_ORIGINS = ["*"]

# UA per le richieste HTTP
RESOLVER_UA = os.getenv(
    "RESOLVER_UA",
    "Mozilla/5.0 (AVD/1.0; +https://ai-video.org)"
)

# Opzioni yt-dlp configurabili da ENV
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

# Domini consentiti per i link
ALLOW_DOMAINS = set([
    "youtube.com", "youtu.be", "vimeo.com",
    "instagram.com", "www.instagram.com",
    "tiktok.com", "www.tiktok.com",
    "facebook.com", "www.facebook.com", "fb.watch",
    "x.com", "twitter.com", "www.twitter.com",
])
# Estensioni file video accettate (download diretto)
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

def _domain_allowed(u: str) -> bool:
    try:
        host = urlparse(u).netloc.lower()
        # consenti anche CDN di file diretti
        if any(host.endswith(d) for d in ALLOW_DOMAINS):
            return True
        # se è un URL diretto con estensione video, consenti
        path = urlparse(u).path.lower()
        if any(path.endswith(ext) for ext in ALLOW_EXTS):
            return True
        return False
    except Exception:
        return False

def _raise_422(msg: str):
    raise HTTPException(status_code=422, detail=msg)

def _tmpfile(suffix: str = "") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path

def _probe_metadata(path: str) -> dict:
    """Estrae width/height/fps via OpenCV se disponibile."""
    meta = {"width": None, "height": None, "fps": None}
    if not HAS_CV2:
        return meta
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

def _basic_analyze(path: str) -> dict:
    """Baseline analyzer: restituisce struttura UI-friendly.
    Integrazioni future possono sostituire questa funzione.
    """
    video_meta = _probe_metadata(path)
    # Placeholder neutro e conservativo per evitare falsi positivi
    result = {
        "label": "uncertain",
        "ai_score": 0.50,        # 0..1 (alto = più probabile AI)
        "confidence": 0.50,      # 0..1
    }
    return {
        "result": result,
        "video": video_meta,
        "timeline": [],
        "timeline_binned": [],
        "peaks": [],
    }

def _save_upload_to_tmp(upload: UploadFile) -> str:
    # controllo dimensione (se disponibile da header)
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
    # reset stream
    try:
        upload.file.close()
    except Exception:
        pass
    return tmp_path

def _download_direct(url: str) -> Tuple[str, Optional[str]]:
    """Prova a scaricare direttamente se è un file video servito come asset."""
    with requests.Session() as s:
        # HEAD per ispezionare headers
        try:
            h = s.head(url, headers=_http_headers(), allow_redirects=True, timeout=REQUEST_TIMEOUT_S)
        except Exception as e:
            _raise_422(f"Errore rete: {e}")

        cl = int(h.headers.get("Content-Length", "0") or "0")
        ctype = h.headers.get("Content-Type", "").lower()
        if cl and cl > RESOLVER_MAX_BYTES:
            _raise_422("Il file remoto è troppo grande. Usa 'Registra 15s' o 'Carica file'.")

        # Se è chiaramente HTML, non è un asset diretto
        if "text/html" in ctype:
            _raise_422("Not a direct video (text/html). Usa 'Carica file' o 'Registra 15s'.")

        # Prova il GET (stream) con limite
        try:
            r = s.get(url, headers=_http_headers(), stream=True, timeout=REQUEST_TIMEOUT_S)
            r.raise_for_status()
        except Exception as e:
            _raise_422(f"Errore download: {e}")

        # Estensione da url o da content-type
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
                if not chunk:
                    break
                size += len(chunk)
                if size > RESOLVER_MAX_BYTES:
                    f.close()
                    try: os.remove(tmp_path)
                    except Exception: pass
                    _raise_422("Dimensione oltre il limite consentito. Usa 'Registra 15s' o 'Carica file'.")
                f.write(chunk)

    return tmp_path, ctype or None

def _ytdlp_resolve_and_download(url: str) -> str:
    """Usa yt-dlp per ottenere il 'miglior' video pubblico e scaricarlo in tmp."""
    if not HAS_YTDLP:
        _raise_422("yt-dlp non disponibile nel server.")

    opts = {**DEFAULT_YTDLP_OPTS, **YTDLP_OPTS}
    # Qui scarichiamo in un temp dir
    tmpdir = tempfile.mkdtemp()
    outtmpl = os.path.join(tmpdir, "%(title).80s.%(ext)s")
    opts.update({
        "outtmpl": outtmpl,
        "restrictfilenames": True,
        "nopart": True,
        "ignoreerrors": False,
        "user_agent": RESOLVER_UA,
        "merge_output_format": "mp4",  # se serve
    })

    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if not info:
                _raise_422("Impossibile estrarre il video (yt-dlp).")
            # Normalizza path del file scaricato (singolo video)
            if "requested_downloads" in info and info["requested_downloads"]:
                fpath = info["requested_downloads"][0]["filepath"]
            else:
                # fallback: prova a comporre dal template
                title = info.get("title", "video")
                ext = info.get("ext", "mp4")
                fpath = os.path.join(tmpdir, f"{title}.{ext}")
            if not os.path.exists(fpath):
                # come extrema ratio, prendi il primo file nel tmpdir
                cand = [os.path.join(tmpdir, p) for p in os.listdir(tmpdir)]
                cand = [p for p in cand if os.path.isfile(p)]
                if not cand:
                    _raise_422("Download non riuscito (yt-dlp).")
                fpath = sorted(cand, key=lambda p: -os.path.getsize(p))[0]
            # Se supera il nostro limite hard, scarta
            if os.path.getsize(fpath) > RESOLVER_MAX_BYTES:
                try: shutil.rmtree(tmpdir)
                except Exception: pass
                _raise_422("File troppo grande dal provider. Usa 'Registra 15s' o 'Carica file'.")
            # Sposta in file tmp definitivo
            suffix = os.path.splitext(fpath)[1] or ".mp4"
            final_path = _tmpfile(suffix=suffix)
            shutil.move(fpath, final_path)
            try: shutil.rmtree(tmpdir)
            except Exception: pass
            return final_path
    except Exception as e:
        # Messaggi noti: login required / rate-limit
        msg = str(e)
        if "login" in msg.lower() and "require" in msg.lower():
            _raise_422("DownloadError yt-dlp: login required / contenuto non pubblico")
        if "rate" in msg.lower() and "limit" in msg.lower():
            _raise_422("DownloadError yt-dlp: rate-limit reached dal provider")
        # generico
        _raise_422(f"DownloadError yt-dlp: {msg}")

# =========================
# Endpoints
# =========================
@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")

@app.post("/analyze")
def analyze(file: UploadFile = File(...)):
    """Analizza un file caricato."""
    if not file:
        _raise_422("Nessun file fornito.")
    try:
        tmp_path = _save_upload_to_tmp(file)
        out = _basic_analyze(tmp_path)
        try: os.remove(tmp_path)
        except Exception: pass
        return JSONResponse(out)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Errore interno: {e}")

@app.post("/analyze-url")
def analyze_url(url: str = Form(None), payload: Optional[dict] = None):
    """Analizza un URL (atteso JSON {url:...} oppure form field 'url')."""
    # Supporta application/json {url:"..."}
    if payload and isinstance(payload, dict) and "url" in payload:
        url_in = payload.get("url")
    else:
        # Se inviato come JSON puro
        # FastAPI mappa automaticamente il body in param 'payload', ma preferiamo leggere raw
        try:
            from fastapi import Request  # lazy
            # NOTA: FastAPI non consente due body parser diversi in una singola sig.
            # Lascio compat: se 'url' non arriva come Form, leggiamo sincronicamente la var env.
        except Exception:
            pass
        url_in = url

    if not url_in:
        # prova a leggere dal body JSON manualmente (retro-compat)
        from starlette.requests import Request
        import anyio
        async def _read_json():
            scope = app.router.default
            return {}
        # più semplice: errore chiaro
        _raise_422("Campo 'url' mancante.")

    url_in = str(url_in).strip()
    if not _is_url(url_in):
        _raise_422("URL non valido.")

    # Se non è dominio consentito e non è asset video diretto, rifiuta
    parsed = urlparse(url_in)
    host = parsed.netloc.lower()
    path = parsed.path.lower()

    is_direct = any(path.endswith(ext) for ext in ALLOW_EXTS)

    if (host not in ALLOW_DOMAINS) and (not is_direct):
        _raise_422("Dominio non supportato. Usa 'Carica file' o 'Registra 15s'.")

    try:
        if is_direct:
            # asset diretto
            tmp_path, ctype = _download_direct(url_in)
        else:
            # social provider → prova yt-dlp se abilitato
            if not USE_YTDLP:
                _raise_422("Richiede estrazione dal provider. Attiva USE_YTDLP o usa 'Carica file'/'Registra 15s'.")
            tmp_path = _ytdlp_resolve_and_download(url_in)

        out = _basic_analyze(tmp_path)
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
    """Retro-compat: accetta file OPPURE url e delega alle route nuove."""
    if file is not None:
        return analyze(file=file)
    if url:
        return analyze_url(url=url)
    _raise_422("Fornire 'file' oppure 'url'.")

# =========================
# Run (sviluppo)
# =========================
# Esempio: uvicorn api:app --host 0.0.0.0 --port 8000
