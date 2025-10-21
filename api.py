import os
import io
import shutil
import tempfile
import traceback
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# --- Config da ENV ---
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))  # 50MB
RESOLVER_MAX_BYTES = int(os.getenv("RESOLVER_MAX_BYTES", str(120 * 1024 * 1024)))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
REQUEST_TIMEOUT_S = int(os.getenv("REQUEST_TIMEOUT_S", "120"))

# --- Import calibrazione/pipeline con fallback ---
try:
    from calibration import calibrate as _calibrate  # type: ignore
except Exception:
    def _calibrate(x: float):
        return float(x), 0.5

try:
    from calibration import combine_scores as _combine_scores  # type: ignore
except Exception:
    def _combine_scores(d: dict):
        vals = [float(v) for v in d.values() if v is not None]
        return (sum(vals) / len(vals)) if vals else 0.5, d

# --- Detectors (fallback sicuri se mancano) ---
try:
    from app.detectors.metadata import ffprobe, score_metadata  # type: ignore
except Exception:
    def ffprobe(path: str) -> dict:
        return {}
    def score_metadata(meta: dict) -> float:
        return 0.5

try:
    from app.detectors.frame_artifacts import score_frame_artifacts  # type: ignore
except Exception:
    def score_frame_artifacts(path: str) -> float:
        return 0.5

try:
    from app.detectors.audio import score_audio  # type: ignore
except Exception:
    def score_audio(path: str) -> float:
        return 0.5

# Resolver URL social/YouTube
try:
    import yt_dlp  # type: ignore
except Exception:
    yt_dlp = None

app = FastAPI(title="AI Video Detector")

# --- CORS ---
if ALLOWED_ORIGINS == "*":
    allow_origins = ["*"]
else:
    allow_origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# -------- Helpers ------------------------------------------------------------

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
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
                raise HTTPException(status_code=413, detail=f"File too large (> {MAX_UPLOAD_BYTES} bytes)")
            out.write(chunk)
    return temp_path

def _download_url_to_temp(url: str) -> str:
    """
    1) Prova con yt-dlp (YouTube/social) usando client 'android'/'web' per aggirare il gate.
    2) Se presente YTDLP_COOKIES_B64, carica i cookie (Netscape) e li passa a yt-dlp.
    3) Se l'URL è diretto a file (.mp4/.webm/.mkv/.mov), usa fallback HTTP.
    """
    import base64
    def _http_fallback(u: str) -> str:
        import requests
        r = requests.get(u, stream=True, timeout=25, headers={
            "User-Agent": "Mozilla/5.0 (Linux; Android 10; Pixel 3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Mobile Safari/537.36"
        })
        if r.status_code != 200:
            raise HTTPException(status_code=422, detail=f"Download HTTP fallito (status {r.status_code})")
        ctype = (r.headers.get("Content-Type") or "").lower()
        looks_video = ("video" in ctype) or any(u.lower().split("?",1)[0].endswith(ext) for ext in (".mp4",".webm",".mkv",".mov"))
        # anche se non sembra video, proviamo lo stesso: alcuni CDN non mettono Content-Type
        fd, path = tempfile.mkdtemp(prefix="aivideo_"), None
        tmpdir = fd
        try:
            ext = ".mp4" if ".mp4" in u.lower() else (".webm" if ".webm" in u.lower() else (".mkv" if ".mkv" in u.lower() else (".mov" if ".mov" in u.lower() else ".bin")))
            path = os.path.join(tmpdir, f"download{ext}")
            size = 0
            with open(path, "wb") as out:
                for chunk in r.iter_content(chunk_size=1024*512):
                    if not chunk: break
                    size += len(chunk)
                    if size > RESOLVER_MAX_BYTES:
                        raise HTTPException(status_code=413, detail="Il video scaricato supera il limite server (fallback)")
                    out.write(chunk)
            if os.path.getsize(path) == 0:
                raise HTTPException(status_code=422, detail="File vuoto dopo download HTTP")
            return path
        except:
            # cleanup temp dir on failure
            try: shutil.rmtree(tmpdir, ignore_errors=True)
            except: pass
            raise

    # Se URL sembra già un file diretto, prova subito il fallback HTTP
    if any(url.lower().split("?",1)[0].endswith(ext) for ext in (".mp4",".webm",".mkv",".mov")):
        return _http_fallback(url)

    if not yt_dlp:
        raise HTTPException(status_code=500, detail="yt-dlp non installato nel server")

    tmp_dir = tempfile.mkdtemp(prefix="aivideo_")
    outtmpl = os.path.join(tmp_dir, "download.%(ext)s")

    # Formato robusto (preferisci mp4 ma con fallback)
    fmt = "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b"

    # Opzioni yt-dlp con client 'android' + 'web' per aggirare il gate
    ydl_opts = {
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "retries": 2,
        "fragment_retries": 2,
        "ignoreerrors": False,
        "noprogress": True,
        "nocheckcertificate": True,
        "quiet": True,
        "no_warnings": True,
        "nopart": True,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Linux; Android 10; Pixel 3) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Mobile Safari/537.36"
            )
        },
        "socket_timeout": 25,
        "http_chunk_size": 10485760,  # 10 MiB
        "postprocessors": [{"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}],
        "ratelimit": 2_000_000,
        "geo_bypass": True,
        # 👇 Forza client YouTube "android" (con fallback a web) per evitare il prompt "sign in to confirm"
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"]
            }
        },
    }

    # Cookie opzionali via env (Netscape cookies.txt in base64)
    cookie_b64 = os.getenv("YTDLP_COOKIES_B64")
    cookie_file = None
    if cookie_b64:
        try:
            raw = base64.b64decode(cookie_b64)
            fd, cookie_file = tempfile.mkstemp(prefix="cookie_", text=True)
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            ydl_opts["cookiefile"] = cookie_file
        except Exception as e:
            traceback.print_exc()
            # Non bloccare: prosegui senza cookie ma lascia traccia
            pass

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if not info:
                raise HTTPException(status_code=422, detail="Estrazione yt-dlp nulla (no info)")
            if "requested_downloads" in info and info["requested_downloads"]:
                filepath = info["requested_downloads"][0]["filepath"]
            else:
                ext = info.get("ext") or "mp4"
                filepath = outtmpl.replace("%(ext)s", ext)

            if not os.path.exists(filepath):
                raise HTTPException(status_code=422, detail="File non creato da yt-dlp")

            size = os.path.getsize(filepath)
            if size == 0:
                raise HTTPException(status_code=422, detail="File vuoto dopo download")
            if size > RESOLVER_MAX_BYTES:
                raise HTTPException(status_code=413, detail="Il video scaricato supera il limite server")

            return filepath

    except yt_dlp.utils.DownloadError as e:
        # Se YouTube richiede login/bot-check e non hai cookie, spiega come fornirli
        msg = str(e) or ""
        hint = ""
        if "Sign in to confirm you're not a bot" in msg or "Sign in to confirm you’re not a bot" in msg:
            hint = " (YouTube richiede cookie: imposta env YTDLP_COOKIES_B64 con il file cookies.txt in base64)"
        raise HTTPException(status_code=422, detail=f"DownloadError yt-dlp: {msg}{hint}")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=f"Risoluzione URL fallita: {str(e) or 'errore generico'}")
    finally:
        # pulizia file cookie
        if cookie_file:
            try: os.remove(cookie_file)
            except: pass
        # NOTA: non rimuoviamo tmp_dir qui se abbiamo restituito un file dentro; /predict lo cancellerà dopo l'analisi



def _analyze_video(video_path: str) -> dict:
    meta = ffprobe(video_path)
    s_meta = score_metadata(meta)
    s_frame = score_frame_artifacts(video_path)
    s_audio = score_audio(video_path)

    combined, details = _combine_scores({
        "metadata": s_meta,
        "frame_artifacts": s_frame,
        "audio": s_audio,
    })
    ai_score, confidence = _calibrate(float(combined))

    duration = (meta or {}).get("duration_s")
    fps = (meta or {}).get("fps")
    frames = (meta or {}).get("frames")

    return {
        "ai_score": round(float(ai_score), 4),
        "confidence": round(float(confidence), 4),
        "details": {
            "metadata": details.get("metadata", s_meta) if isinstance(details, dict) else s_meta,
            "frame_artifacts": details.get("frame_artifacts", s_frame) if isinstance(details, dict) else s_frame,
            "audio": details.get("audio", s_audio) if isinstance(details, dict) else s_audio,
        },
        "duration_s": duration,
        "fps": fps,
        "frames": frames,
    }

# -------- Routes -------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="it"><head><meta charset="utf-8">
<title>AI Video Detector</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
</head><body style="font-family: system-ui, sans-serif; padding:24px">
<h1>AI Video Detector</h1>
<p>Inserisci un link (YouTube/social) oppure carica un file (max 50MB).</p>
<form method="post" action="/predict" enctype="multipart/form-data">
  <div>
    <label>URL (YouTube/TikTok/X/Instagram/Facebook/Reddit)</label><br>
    <input type="url" name="url" placeholder="https://..." style="width:420px">
  </div>
  <p><em>— oppure —</em></p>
  <div>
    <label>File video</label><br>
    <input type="file" name="file">
  </div>
  <p><button type="submit">Analizza</button></p>
</form>
<p style="opacity:.7">Endpoint GET rapidi: <code>/predict-get?url=...</code> oppure <code>/predict?url=...</code></p>
</body></html>"""

@app.get("/healthz", response_class=PlainTextResponse)
def healthz() -> str:
    return "ok"

@app.get("/predict-get")
@app.get("/predict")
def predict_get(url: Optional[str] = None):
    if not url:
        raise HTTPException(status_code=400, detail="Parametro 'url' mancante")
    path = _download_url_to_temp(url)
    try:
        result = _analyze_video(path)
        return JSONResponse(result)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

@app.post("/predict")
async def predict(url: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
    if not url and not file:
        raise HTTPException(status_code=400, detail="Fornisci 'url' oppure 'file'")
    temp_path = None
    try:
        if url:
            temp_path = _download_url_to_temp(url)
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
            try:
                os.remove(temp_path)
            except Exception:
                pass

# --- Alias per il plugin WordPress ---
@app.post("/analyze")
async def analyze_legacy(url: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
    return await predict(url=url, file=file)

@app.post("/analyze-url")
async def analyze_url_legacy(url: Optional[str] = Form(None)):
    if not url:
        raise HTTPException(status_code=400, detail="Parametro 'url' mancante")
    return await predict(url=url, file=None)
