import os
import re
import json
import tempfile
import subprocess
import urllib.parse
from typing import Optional, Tuple, Dict, Any

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

# ====== Config ======
APP_NAME = "AI-Video Detector"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))       # 50MB
RESOLVER_MAX_BYTES = int(os.getenv("RESOLVER_MAX_BYTES", str(120 * 1024 * 1024)))  # 120MB hard cap
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "6.0"))
READ_TIMEOUT = float(os.getenv("READ_TIMEOUT", "15.0"))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "").strip()
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
USER_AGENT = os.getenv("RESOLVER_UA", "Mozilla/5.0 (AVD/1.0; +https://ai-video.org)")
USE_YTDLP = os.getenv("USE_YTDLP", "0") in ("1", "true", "True", "yes", "YES")
YTDLP_COOKIES = os.getenv("YTDLP_COOKIES", "")

SOCIAL_LOGIN_WALL = ("instagram.com","tiktok.com","facebook.com","x.com","twitter.com","reddit.com")

# ====== App & CORS ======
app = FastAPI(title=APP_NAME)
allow_origins = ALLOWED_ORIGINS or ([FRONTEND_ORIGIN] if FRONTEND_ORIGIN else ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Utils ======
def _is_probably_video(header: bytes) -> bool:
    if b"ftyp" in header[:96]: return True
    if header.startswith(b"\x1A\x45\xDF\xA3"): return True  # EBML (webm/mkv)
    return False

def _looks_like_html(header: bytes) -> bool:
    h = header[:64].lstrip().lower()
    return h.startswith(b"<!doct") or h.startswith(b"<html") or b"<head" in header[:256].lower()

def _is_m3u8(header: bytes, url: str) -> bool:
    if url.lower().endswith(".m3u8"): return True
    return header.startswith(b"#EXTM3U") or (b"#EXTM3U" in header[:1024])

def _hex_preview(b: bytes, n: int = 128) -> str:
    return b[:n].hex(" ", 1)

def _domain(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).hostname or ""
    except Exception:
        return ""

def _is_social_loginwall(url: str) -> bool:
    host = _domain(url).lower()
    return any(host.endswith(d) for d in SOCIAL_LOGIN_WALL)

def _run_ffprobe(path: str) -> Dict[str, Any]:
    cmd = ["ffprobe","-v","error","-hide_banner","-show_streams","-show_format","-of","json",path]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        data = json.loads(res.stdout.strip()) if res.stdout.strip() else {}
        return {"returncode": res.returncode, "stdout": data, "stderr": res.stderr.strip()}
    except Exception as e:
        return {"returncode": -1, "stdout": {}, "stderr": f"ffprobe exec error: {e}"}

def _requests_stream(url: str, limit_bytes: int) -> Tuple[str, int]:
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as r:
            status = r.status_code
            ctype = (r.headers.get("Content-Type") or "").lower()
            if status >= 400:
                raise HTTPException(status_code=422, detail={"error": f"HTTP {status}", "url": url, "content_type": ctype})
            it = r.iter_content(chunk_size=8192)
            try:
                first = next(it)
            except StopIteration:
                first = b""
            header = first or b""
            if _looks_like_html(header):
                raise HTTPException(status_code=415, detail={
                    "error":"URL ha restituito HTML/pagina di login",
                    "url":url,
                    "header_hex_128B":_hex_preview(header),
                    "hint":"Per social protetti: carica file o usa 'Registra 10s'."
                })
            if _is_m3u8(header, url):
                raise HTTPException(status_code=415, detail={
                    "error":"Stream HLS (.m3u8) rilevato",
                    "url":url,
                    "hint":"Registra 10s e carica il file, o fornisci un MP4/WEBM scaricabile."
                })
            total = 0
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=_suffix_from_ctype_or_url(ctype, url))
            tmp_path = tmp.name
            try:
                if header:
                    tmp.write(header); total += len(header)
                for chunk in it:
                    if not chunk: continue
                    total += len(chunk)
                    if total > limit_bytes:
                        raise HTTPException(status_code=413, detail={
                            "error":"File troppo grande dal resolver","max_bytes":limit_bytes,"url":url
                        })
                    tmp.write(chunk)
            finally:
                tmp.close()
            return tmp_path, total
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=422, detail={"error":"Errore di rete durante il download","exc":str(e),"url":url})

def _suffix_from_ctype_or_url(ctype: str, url: str) -> str:
    u = url.lower()
    if "mp4" in ctype or u.endswith(".mp4"): return ".mp4"
    if "webm" in ctype or u.endswith(".webm"): return ".webm"
    if "quicktime" in ctype or u.endswith(".mov"): return ".mov"
    if "x-matroska" in ctype or u.endswith(".mkv"): return ".mkv"
    return ".bin"

def _try_ytdlp(url: str):
    if not USE_YTDLP: return None, "yt-dlp disabilitato"
    args = ["yt-dlp","-g","-f","bv*+ba/best","--no-warnings","--no-call-home",url]
    if YTDLP_COOKIES: args += ["--cookies", YTDLP_COOKIES]
    try:
        res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        if res.returncode != 0:
            return None, f"yt-dlp failed ({res.returncode}): {res.stderr.strip()}"
        direct = res.stdout.strip().splitlines()[-1].strip() if res.stdout.strip() else ""
        return (direct if direct else None), (None if direct else "yt-dlp non ha fornito URL diretto")
    except Exception as e:
        return None, f"yt-dlp exec error: {e}"

# ====== Endpoints ======
@app.get("/", response_class=PlainTextResponse)
def root(): return f"{APP_NAME} up. Endpoints: /healthz, /analyze, /analyze-url, /predict"

@app.get("/healthz", response_class=PlainTextResponse)
def healthz(): return "ok"

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # leggi e NON perdere i primi byte
    header = await file.read(8192)
    if not header: raise HTTPException(status_code=415, detail={"error":"File vuoto o non ricevuto"})
    total = 0
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(header); total += len(header)
        while True:
            chunk = await file.read(1024*1024)
            if not chunk: break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                try: os.remove(tmp_path)
                except: pass
                raise HTTPException(status_code=413, detail={"error":"File troppo grande","max_bytes":MAX_UPLOAD_BYTES,"got":total})
            tmp.write(chunk)

    sniff_warning = None
    if not _is_probably_video(header):
        sniff_warning = "Header non riconosciuto come video (manca 'ftyp' o EBML)"

    probe = _run_ffprobe(tmp_path)
    if not probe.get("stdout", {}).get("streams"):
        try: os.remove(tmp_path)
        except: pass
        raise HTTPException(status_code=415, detail={
            "error":"Unsupported or empty media: no audio/video streams found",
            "ffprobe_found":True,
            "ffprobe_returncode":probe.get("returncode"),
            "ffprobe_stderr":probe.get("stderr"),
            "file_size_written": total,
            "header_hex_128B": _hex_preview(header),
            "sniff_warning": sniff_warning,
            "hint":"Verifica che sia un vero MP4/WEBM/MOV."
        })

    # === ANALISI ===
    from app.analyzers import run as analyze_media
    result = analyze_media(tmp_path, ffprobe_json=probe["stdout"])

    try: os.remove(tmp_path)
    except: pass
    return JSONResponse(result)

@app.post("/analyze-url")
def analyze_url(url: str = Form(...)):
    if not re.match(r"^https?://", url, flags=re.I):
        raise HTTPException(status_code=422, detail={"error":"URL non valido","url":url})
    resolved = url
    if _is_social_loginwall(url):
        direct, err = _try_ytdlp(url)
        if direct:
            resolved = direct
        else:
            raise HTTPException(status_code=422, detail={
                "error":"Contenuto protetto / login richiesto",
                "url":url,"ytdlp_error":err or "yt-dlp non disponibile",
                "hint":"Per Instagram/TikTok: 'Carica file' o 'Registra 10s'."
            })
    tmp_path, total = _requests_stream(resolved, RESOLVER_MAX_BYTES)
    probe = _run_ffprobe(tmp_path)
    if not probe.get("stdout", {}).get("streams"):
        try:
            with open(tmp_path,"rb") as f: header = f.read(128)
        except: header = b""
        finally:
            try: os.remove(tmp_path)
            except: pass
        raise HTTPException(status_code=415, detail={
            "error":"Unsupported or empty media",
            "ffprobe_returncode": probe.get("returncode"),
            "ffprobe_stderr": probe.get("stderr"),
            "downloaded_bytes": total,
            "header_hex_128B": _hex_preview(header),
            "source_url": url,
            "resolved_url": resolved,
            "hint":"Fornisci un MP4/WEBM o registra 10s."
        })

    from app.analyzers import run as analyze_media
    result = analyze_media(tmp_path, source_url=url, resolved_url=resolved, ffprobe_json=probe["stdout"])
    try: os.remove(tmp_path)
    except: pass
    return JSONResponse(result)

@app.post("/predict")
async def predict(file: Optional[UploadFile] = File(None), url: Optional[str] = Form(None)):
    if file is not None: return await analyze(file=file)
    if url: return analyze_url(url=url)
    raise HTTPException(status_code=422, detail={"error":"Specificare 'file' oppure 'url'."})