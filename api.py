import os
import io
import re
import json
import shutil
import tempfile
import subprocess
import urllib.parse
from typing import Optional, Tuple, Dict, Any

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

# =========================
# Config da ENV (default sicuri)
# =========================
APP_NAME = "AI-Video Detector"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))       # 50MB
RESOLVER_MAX_BYTES = int(os.getenv("RESOLVER_MAX_BYTES", str(120 * 1024 * 1024)))  # 120MB hard cap downloader
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "6.0"))
READ_TIMEOUT = float(os.getenv("READ_TIMEOUT", "15.0"))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "").strip()
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
USER_AGENT = os.getenv("RESOLVER_UA", "Mozilla/5.0 (AVD/0.5; +https://ai-video.org)")
USE_YTDLP = os.getenv("USE_YTDLP", "0") in ("1", "true", "True", "yes", "YES")
YTDLP_COOKIES = os.getenv("YTDLP_COOKIES", "")  # path a cookies.txt opzionale

# Dominî social “difficili” → suggerisci upload o 10s record
SOCIAL_LOGIN_WALL = (
    "instagram.com",
    "tiktok.com",
    "facebook.com",
    "x.com",
    "twitter.com",
    "reddit.com",
)

# =========================
# App & CORS
# =========================
app = FastAPI(title=APP_NAME)

allow_origins = ALLOWED_ORIGINS or ([FRONTEND_ORIGIN] if FRONTEND_ORIGIN else ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Utils
# =========================
def _is_probably_video(header: bytes) -> bool:
    # mp4/mov: 'ftyp' nelle prime decine di byte; webm/mkv: EBML 0x1A45DFA3
    if b"ftyp" in header[:96]:
        return True
    if header.startswith(b"\x1A\x45\xDF\xA3"):
        return True
    # stream .mp4 frammentati possono avere box size prima di 'ftyp'
    return False

def _looks_like_html(header: bytes) -> bool:
    h = header[:64].lstrip()
    return h.startswith(b"<!DOCT") or h.startswith(b"<html") or b"<head" in header[:256].lower()

def _is_m3u8(header: bytes, url: str) -> bool:
    if url.lower().endswith(".m3u8"):
        return True
    # firma testuale HLS
    if header.startswith(b"#EXTM3U") or b"#EXTM3U" in header[:1024]:
        return True
    return False

def _run_ffprobe(path: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error",
        "-hide_banner",
        "-show_streams", "-show_format",
        "-of", "json", path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        out = res.stdout.strip()
        err = res.stderr.strip()
        data = json.loads(out) if out else {}
        return {"returncode": res.returncode, "stdout": data, "stderr": err}
    except Exception as e:
        return {"returncode": -1, "stdout": {}, "stderr": f"ffprobe exec error: {e}"}

def _domain(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).hostname or ""
    except Exception:
        return ""

def _is_social_loginwall(url: str) -> bool:
    host = _domain(url).lower()
    return any(host.endswith(d) for d in SOCIAL_LOGIN_WALL)

def _hex_preview(b: bytes, n: int = 128) -> str:
    return b[:n].hex(" ", 1)

def _requests_stream(url: str, limit_bytes: int) -> Tuple[str, int]:
    """
    Scarica (fino a limit_bytes) in un file temporaneo.
    Ritorna (path, total_written).
    Solleva HTTPException con diagnostica in caso di problemi.
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as r:
            status = r.status_code
            ctype = (r.headers.get("Content-Type") or "").lower()
            clen = r.headers.get("Content-Length")
            if status >= 400:
                raise HTTPException(status_code=422, detail={
                    "error": f"Download non riuscito: HTTP {status}",
                    "url": url,
                    "content_type": ctype,
                })
            # leggi un po' per sniff
            it = r.iter_content(chunk_size=8192)
            try:
                first = next(it)
            except StopIteration:
                first = b""
            header = first or b""
            if _looks_like_html(header):
                raise HTTPException(status_code=415, detail={
                    "error": "URL ha restituito HTML/pagina di login, non un video",
                    "url": url,
                    "header_hex_128B": _hex_preview(header),
                    "hint": "Se è Instagram/TikTok ecc. carica il file o usa 'Registra 10s'.",
                })
            if _is_m3u8(header, url):
                raise HTTPException(status_code=415, detail={
                    "error": "Stream HLS (.m3u8) rilevato",
                    "url": url,
                    "hint": "Registra 10s e carica il file, oppure fornisci un MP4/WEBM scaricabile.",
                })

            # scrivi su disco rispettando il limite
            total = 0
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=_suffix_from_ctype_or_url(ctype, url))
            tmp_path = tmp.name
            try:
                if header:
                    tmp.write(header)
                    total += len(header)
                for chunk in it:
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > limit_bytes:
                        raise HTTPException(status_code=413, detail={
                            "error": "File troppo grande dal resolver",
                            "max_bytes": limit_bytes,
                            "url": url,
                            "hint": "Fornisci un file <= 50MB oppure usa 'Registra 10s'."
                        })
                    tmp.write(chunk)
            finally:
                tmp.close()
            return tmp_path, total
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=422, detail={
            "error": "Errore di rete durante il download",
            "exc": str(e),
            "url": url
        })

def _suffix_from_ctype_or_url(ctype: str, url: str) -> str:
    if "mp4" in ctype or url.lower().endswith(".mp4"):
        return ".mp4"
    if "webm" in ctype or url.lower().endswith(".webm"):
        return ".webm"
    if "quicktime" in ctype or url.lower().endswith(".mov") or "x-matroska" in ctype or url.lower().endswith(".mkv"):
        # mov/mkv: comunque analizziamo con ffprobe
        return ".mov" if ".mov" in url.lower() or "quicktime" in ctype else ".mkv"
    return ".bin"

def _try_ytdlp(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Prova a risolvere l'URL con yt-dlp, restituendo (direct_url, err).
    Se USE_YTDLP è falso, ritorna (None, 'yt-dlp disabilitato').
    """
    if not USE_YTDLP:
        return None, "yt-dlp disabilitato"
    # costruisci comando
    # Strategia: estrai URL "best" senza scaricare (simulate JSON), poi lo useremo con requests per limitare byte.
    # In alternativa, puoi scaricare direttamente con -o tmpfile, ma qui preferiamo URL diretto.
    args = ["yt-dlp", "-g", "-f", "bv*+ba/best", "--no-warnings", "--no-call-home", url]
    if YTDLP_COOKIES:
        args.extend(["--cookies", YTDLP_COOKIES])
    try:
        res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        if res.returncode != 0:
            return None, f"yt-dlp failed ({res.returncode}): {res.stderr.strip()}"
        direct_url = res.stdout.strip().splitlines()[-1].strip() if res.stdout.strip() else ""
        if not direct_url:
            return None, "yt-dlp non ha fornito un URL video diretto"
        return direct_url, None
    except Exception as e:
        return None, f"yt-dlp exec error: {e}"

# =========================
# Endpoints
# =========================

@app.get("/", response_class=PlainTextResponse)
def root():
    return f"{APP_NAME} backend is up. Endpoints: /healthz, /analyze, /analyze-url, /analyze-link, /predict"

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Analizza un file caricato. Scrive i primi byte PRIMA sul disco per non perderli.
    """
    header = await file.read(8192)
    if not header:
        raise HTTPException(status_code=415, detail={"error": "File vuoto o non ricevuto"})

    total_written = 0
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(header)
        total_written += len(header)
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total_written += len(chunk)
            if total_written > MAX_UPLOAD_BYTES:
                try: os.remove(tmp_path)
                except: pass
                raise HTTPException(status_code=413, detail={
                    "error": "File troppo grande",
                    "max_bytes": MAX_UPLOAD_BYTES,
                    "got": total_written
                })
            tmp.write(chunk)

    sniff_warning = None
    if not _is_probably_video(header):
        sniff_warning = "Header non riconosciuto come video (manca 'ftyp' o EBML)"

    probe = _run_ffprobe(tmp_path)
    has_streams = bool(probe.get("stdout", {}).get("streams"))

    if not has_streams:
        try: size_on_disk = os.path.getsize(tmp_path)
        except: size_on_disk = None
        try: os.remove(tmp_path)
        except: pass
        detail = {
            "error": "Unsupported or empty media: no audio/video streams found",
            "ffprobe_found": True,
            "ffprobe_returncode": probe.get("returncode"),
            "ffprobe_stderr": probe.get("stderr"),
            "file_size_written": total_written,
            "size_on_disk": size_on_disk,
            "header_hex_128B": _hex_preview(header),
            "sniff_warning": sniff_warning,
            "hint": "Se è un URL social usa /analyze-url; se è un upload, verifica che sia un vero MP4/WEBM/MOV."
        }
        raise HTTPException(status_code=415, detail=detail)

    fmt = probe["stdout"].get("format", {})
    streams = probe["stdout"].get("streams", [])
    try: os.remove(tmp_path)
    except: pass

    return JSONResponse({
        "ok": True,
        "meta": {
            "filename": file.filename,
            "size_bytes": total_written,
            "format": {
                "format_name": fmt.get("format_name"),
                "duration": fmt.get("duration"),
                "bit_rate": fmt.get("bit_rate")
            },
            "streams_summary": [{"codec_type": s.get("codec_type"), "codec_name": s.get("codec_name")} for s in streams]
        }
    })

@app.post("/analyze-url")
def analyze_url(url: str = Form(...)):
    """
    Risolve un URL e analizza il media scaricato (entro RESOLVER_MAX_BYTES).
    - Rileva HTML/login e HLS (.m3u8).
    - Se abilitato, prova yt-dlp per social.
    """
    if not re.match(r"^https?://", url, flags=re.I):
        raise HTTPException(status_code=422, detail={"error": "URL non valido", "url": url})

    # Se social “login-wall”, prova yt-dlp se consentito
    direct_url = None
    ytdlp_err = None
    if _is_social_loginwall(url):
        direct_url, ytdlp_err = _try_ytdlp(url)
        if direct_url:
            url_to_fetch = direct_url
        else:
            # niente yt-dlp o errore: messaggio guidato alle tue preferenze UI
            raise HTTPException(status_code=422, detail={
                "error": "Contenuto protetto / login richiesto",
                "url": url,
                "ytdlp_error": ytdlp_err or "yt-dlp non disponibile",
                "hint": "Per Instagram/TikTok ecc.: usa 'Carica file' oppure 'Registra 10s'."
            })
    else:
        url_to_fetch = url

    tmp_path, total = _requests_stream(url_to_fetch, RESOLVER_MAX_BYTES)

    # ffprobe
    probe = _run_ffprobe(tmp_path)
    has_streams = bool(probe.get("stdout", {}).get("streams"))

    if not has_streams:
        try: size_on_disk = os.path.getsize(tmp_path)
        except: size_on_disk = None
        try: 
            with open(tmp_path, "rb") as f:
                header = f.read(128)
        except:
            header = b""
        finally:
            try: os.remove(tmp_path)
            except: pass

        detail = {
            "error": "Unsupported or empty media: no audio/video streams found",
            "ffprobe_found": True,
            "ffprobe_returncode": probe.get("returncode"),
            "ffprobe_stderr": probe.get("stderr"),
            "downloaded_bytes": total,
            "size_on_disk": size_on_disk,
            "header_hex_128B": _hex_preview(header),
            "source_url": url,
            "resolved_url": url_to_fetch,
            "hint": "Il link potrebbe non essere un file video diretto. Fornisci un MP4/WEBM o usa 'Registra 10s'."
        }
        raise HTTPException(status_code=415, detail=detail)

    fmt = probe["stdout"].get("format", {})
    streams = probe["stdout"].get("streams", [])
    try: os.remove(tmp_path)
    except: pass

    return JSONResponse({
        "ok": True,
        "meta": {
            "source_url": url,
            "resolved_url": url_to_fetch,
            "downloaded_bytes": total,
            "format": {
                "format_name": fmt.get("format_name"),
                "duration": fmt.get("duration"),
                "bit_rate": fmt.get("bit_rate")
            },
            "streams_summary": [{"codec_type": s.get("codec_type"), "codec_name": s.get("codec_name")} for s in streams]
        }
    })

@app.post("/analyze-link")
def analyze_link(url: str = Form(...)):
    """
    Analisi 'leggera' del link: HEAD/GET parziale per capire se sembra video,
    senza scaricare l'intero file.
    """
    if not re.match(r"^https?://", url, flags=re.I):
        raise HTTPException(status_code=422, detail={"error": "URL non valido", "url": url})

    headers = {"User-Agent": USER_AGENT, "Accept": "*/*", "Range": "bytes=0-8191"}
    try:
        r = requests.get(url, headers=headers, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=422, detail={"error": "Errore di rete", "exc": str(e), "url": url})

    ctype = (r.headers.get("Content-Type") or "").lower()
    status = r.status_code
    if status >= 400:
        raise HTTPException(status_code=422, detail={"error": f"HTTP {status}", "content_type": ctype, "url": url})

    content = b""
    try:
        for chunk in r.iter_content(8192):
            content += chunk or b""
            break
    except requests.exceptions.RequestException:
        pass

    if _looks_like_html(content):
        raise HTTPException(status_code=415, detail={
            "error": "HTML/pagina di login",
            "content_type": ctype,
            "url": url,
            "hint": "Per social protetti carica il file o usa 'Registra 10s'."
        })
    if _is_m3u8(content, url):
        raise HTTPException(status_code=415, detail={
            "error": "HLS (.m3u8) rilevato",
            "url": url,
            "hint": "Registra 10s del video e carica il file."
        })

    return JSONResponse({
        "ok": True,
        "link_probe": {
            "url": url,
            "content_type": ctype,
            "looks_video_header": _is_probably_video(content),
            "header_hex_128B": _hex_preview(content),
        }
    })

@app.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
):
    """
    Retro-compatibilità: accetta file o url. Se entrambi, preferisce il file.
    """
    if file is not None:
        return await analyze(file=file)
    if url:
        return analyze_url(url=url)
    raise HTTPException(status_code=422, detail={"error": "Specificare 'file' oppure 'url'."})