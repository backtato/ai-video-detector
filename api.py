# api.py
# FastAPI backend per AI-Video Detector – versione compat-max
# Endpoint: /, /healthz, /version, /analyze, /analyze-url, /predict, /debug/resolver, /debug/ffprobe
# Pipeline: download (yt-dlp/httpx+cache) → ffprobe meta → forensic (ExifTool/C2PA) → analyzers (video/audio/heuristics) → fusion
# Non rimuove funzionalità esistenti, aggiunge forensics, caching e diagnostica.

import os
import io
import re
import json
import uuid
import shutil
import tempfile
import traceback
import subprocess
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# ==== Config ENV ====
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))   # 50 MB
RESOLVER_MAX_BYTES = int(os.getenv("RESOLVER_MAX_BYTES", str(120 * 1024 * 1024)))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
USE_YTDLP = os.getenv("USE_YTDLP", "1").lower() not in ("0", "false", "no")
CACHE_DIR = os.environ.get("CACHE_DIR", "/tmp/aivideo-cache")
APP_VERSION = os.getenv("APP_VERSION", "1.1.3-compatmax")

# ==== Disk cache (opzionale) ====
try:
    from diskcache import Cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = Cache(CACHE_DIR)
except Exception:
    cache = None

# ==== FastAPI app & CORS ====
app = FastAPI(title="AI-Video Detector API", version=APP_VERSION)
if ALLOWED_ORIGINS == "*" or not ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"]
    )
else:
    origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware, allow_origins=origins, allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"]
    )

# ==== Local imports (pipeline) ====
from app.analyzers import video as video_mod
from app.analyzers import audio as audio_mod
from app.analyzers import heuristics_v2
try:
    from app.analyzers import forensic as forensic_mod
except Exception:
    forensic_mod = None
from app import fusion as fusion_mod

# ==== Utils ====
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

def _tmpdir(prefix="aiv_") -> str:
    d = tempfile.mkdtemp(prefix=prefix)
    return d

def _tmpfile(suffix: str) -> str:
    d = _tmpdir()
    return os.path.join(d, f"{uuid.uuid4().hex}{suffix}")

def _is_html(content_type: str, text_head: bytes) -> bool:
    ct = (content_type or "").lower()
    if "text/html" in ct:
        return True
    head = (text_head or b"")[:4096].strip().lower()
    return head.startswith(b"<!doctype html") or b"<html" in head

def _seems_hls(url: str) -> bool:
    u = (url or "").lower()
    return (".m3u8" in u) or (".m3u" in u)

def _ffprobe_summary(path: str) -> Dict[str, Any]:
    try:
        cmd = ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-print_format", "json", path]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        data = json.loads(p.stdout or "{}")
        streams = data.get("streams", [])
        fmt = data.get("format", {})
        width = height = 0
        fps = 0.0
        duration = 0.0
        bit_rate = 0.0
        vcodec = acodec = None

        for s in streams:
            if s.get("codec_type") == "video":
                width = int(s.get("width") or width or 0)
                height = int(s.get("height") or height or 0)
                avg = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/0"
                try:
                    num, den = avg.split("/")
                    num, den = float(num), float(den)
                    fps = float(num / den) if den > 0 else fps
                except Exception:
                    pass
                vcodec = s.get("codec_name") or vcodec
                try:
                    d = float(s.get("duration"))
                    duration = d if d > 0 else duration
                except Exception:
                    pass
            elif s.get("codec_type") == "audio":
                acodec = s.get("codec_name") or acodec

        try:
            if not duration:
                duration = float(fmt.get("duration", 0.0))
        except Exception:
            pass
        try:
            bit_rate = float(fmt.get("bit_rate", 0.0))
        except Exception:
            bit_rate = 0.0

        return {
            "width": width, "height": height,
            "fps": fps, "duration": duration, "bit_rate": bit_rate,
            "vcodec": vcodec, "acodec": acodec,
            "format_name": fmt.get("format_name")
        }
    except Exception as e:
        return {"error": f"ffprobe_error: {e}"}

# ==== Resolver / Downloader ====
async def _download_via_httpx(url: str, max_bytes: int = RESOLVER_MAX_BYTES) -> Dict[str, Any]:
    key = f"httpx:{url}"
    if cache:
        cached = cache.get(key)
        if cached and os.path.exists(cached):
            return {"ok": True, "path": cached, "platform": "direct", "warnings": ["cache_hit"]}

    async with httpx.AsyncClient(follow_redirects=True, timeout=25.0) as client:
        r = await client.get(url, headers={"User-Agent": UA})
        ct = r.headers.get("content-type", "")
        head = r.content[:4096]
        if _is_html(ct, head):
            return {"ok": False, "error": "html_response", "content_type": ct}
        if _seems_hls(url):
            return {"ok": False, "error": "hls_stream"}

        total = int(r.headers.get("content-length") or 0)
        if total and total > max_bytes:
            return {"ok": False, "error": "too_large", "size": total}

        dst = _tmpfile(".bin")
        read = 0
        with open(dst, "wb") as f:
            async for chunk in r.aiter_bytes():
                read += len(chunk)
                if read > max_bytes:
                    f.close()
                    try: os.remove(dst)
                    except Exception: pass
                    return {"ok": False, "error": "too_large_stream", "size": read}
                f.write(chunk)

    if cache:
        cache.set(key, dst, expire=7200)
    return {"ok": True, "path": dst, "platform": "direct", "warnings": []}

async def _download_via_ytdlp(url: str) -> Dict[str, Any]:
    if not USE_YTDLP:
        return {"ok": False, "error": "ytdlp_disabled"}
    key = f"ytdlp:{url}"
    if cache:
        cached = cache.get(key)
        if cached and os.path.exists(cached):
            return {"ok": True, "path": cached, "platform": "yt-dlp", "warnings": ["cache_hit"]}

    dst = _tmpfile(".mp4")
    cmd = [
        "yt-dlp",
        "-f", "bv*+ba/b",
        "-o", dst,
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--no-warnings",
        url
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if p.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        return {
            "ok": False,
            "error": "ytdlp_error",
            "stderr": p.stderr[-1000:],
            "hints": {
                "needs_cookies": True,
                "login_required_or_ratelimit": True,
                "advice": "Link protetto/rate-limited: usa 'Carica file' o 'Registra 10s e carica'."
            }
        }
    if cache:
        cache.set(key, dst, expire=7200)
    return {"ok": True, "path": dst, "platform": "yt-dlp", "warnings": []}

def _resolver_summary(url: str) -> Dict[str, Any]:
    # Riepilogo semplice per /debug/resolver
    info = {
        "input_url": url,
        "is_hls_like": _seems_hls(url),
        "use_ytdlp": USE_YTDLP,
        "cache_enabled": cache is not None
    }
    return info

# ==== Analisi core ====
def _analyze_file(path: str, source_url: Optional[str] = None, platform: Optional[str] = None) -> Dict[str, Any]:
    # 0) Meta base via ffprobe
    meta = _ffprobe_summary(path)
    meta["source_url"] = source_url
    meta["resolved_url"] = None

    # 0b) Forensics
    forensic = {}
    if forensic_mod and hasattr(forensic_mod, "analyze"):
        try:
            forensic = forensic_mod.analyze(path)
        except Exception as e:
            forensic = {"error": f"forensic_error: {e}"}
    meta["forensic"] = forensic

    # 1) Video
    vstats = video_mod.analyze(path)

    # 2) Audio
    try:
        astats = audio_mod.analyze(path)
    except Exception as e:
        astats = {"error": f"audio_error: {e}"}

    # 3) Heuristics (usa vstats/astats/meta completi)
    hints = heuristics_v2.compute_hints(vstats, astats, meta)

    # 4) Fusione
    fusion_out = fusion_mod.fuse(video_stats=vstats, audio_stats=astats, hints=hints, meta=meta)

    # 5) Meta per output (compat UI)
    meta_block = {
        "width": meta.get("width"),
        "height": meta.get("height"),
        "fps": meta.get("fps"),
        "duration": meta.get("duration"),
        "bit_rate": meta.get("bit_rate"),
        "vcodec": meta.get("vcodec"),
        "acodec": meta.get("acodec"),
        "format_name": meta.get("format_name"),
        "source_url": source_url,
        "resolved_url": meta.get("resolved_url"),
        "forensic": forensic
    }

    out = {
        "ok": True,
        "meta": meta_block,
        "video": vstats,
        "audio": astats,
        "hints": hints,
        **fusion_out
    }
    return out

# ==== Root & Health ====
@app.get("/")
def root():
    return HTMLResponse(
        f"<h1>AI-Video Detector</h1>"
        f"<p>Version: {APP_VERSION}</p>"
        f"<ul>"
        f"<li>POST /analyze (file upload)</li>"
        f"<li>POST /analyze-url (url|q|link)</li>"
        f"<li>POST /predict (retro-compat)</li>"
        f"<li>GET /healthz</li>"
        f"<li>GET /version</li>"
        f"<li>GET /debug/resolver?url=...</li>"
        f"<li>GET /debug/ffprobe?path=/absolute/path</li>"
        f"</ul>"
    )

@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")

@app.get("/version")
def version():
    cfg = {
        "MAX_UPLOAD_BYTES": MAX_UPLOAD_BYTES,
        "RESOLVER_MAX_BYTES": RESOLVER_MAX_BYTES,
        "USE_YTDLP": USE_YTDLP,
        "CACHE_DIR": CACHE_DIR,
        "ALLOWED_ORIGINS": ALLOWED_ORIGINS
    }
    return JSONResponse({"version": APP_VERSION, "config": cfg})

# ==== Endpoints diagnostici (facoltativi ma utili) ====
@app.get("/debug/resolver")
async def debug_resolver(url: str = Query(..., description="URL da sondare")):
    info = _resolver_summary(url)
    # Non effettua download, solo riepilogo
    return JSONResponse(info)

@app.get("/debug/ffprobe")
def debug_ffprobe(path: str = Query(..., description="Path assoluto file locale")):
    if not os.path.exists(path):
        raise HTTPException(404, detail="File non trovato")
    return JSONResponse(_ffprobe_summary(path))

# ==== Analyze via upload ====
@app.post("/analyze")
async def analyze(file: UploadFile = File(None), url: Optional[str] = Form(None)):
    """
    Caricamento file (consigliato). Per URL, preferisci /analyze-url.
    Compat: accetta anche 'url' dentro form e reindirizza.
    """
    if file and file.filename:
        content = await file.read()
        if not content:
            raise HTTPException(415, detail={"error": "File vuoto o non ricevuto"})
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, detail={"error": "File troppo grande", "max_bytes": MAX_UPLOAD_BYTES})
        suffix = os.path.splitext(file.filename)[1] or ".bin"
        dst = _tmpfile(suffix)
        with open(dst, "wb") as f:
            f.write(content)
        try:
            return JSONResponse(_analyze_file(dst, source_url=None, platform="upload"))
        finally:
            # eventuale cleanup differito (lasciamo i tmp per debug)
            pass

    if url:
        # fallback: trattiamo come analyze-url
        return await analyze_url(url=url)

    raise HTTPException(415, detail={"error": "File vuoto o non ricevuto"})

# ==== Analyze via URL (yt-dlp + fallback httpx) ====
@app.post("/analyze-url")
async def analyze_url(
    url: Optional[str] = Form(None),
    q: Optional[str] = Form(None),
    link: Optional[str] = Form(None),
):
    """
    Accetta url anche come 'q' o 'link' (retro-compat).
    Prova yt-dlp se abilitato; in caso di contenuto protetto/ratelimit → 422 con hint.
    Fallback httpx (blocca HLS/HTML).
    """
    url = url or q or link
    if not url:
        raise HTTPException(422, detail="URL mancante")

    via = None
    if USE_YTDLP:
        y = await _download_via_ytdlp(url)
        if y.get("ok"):
            via = y
        else:
            if y.get("hints", {}).get("login_required_or_ratelimit"):
                # Messaggio esplicito per UI
                raise HTTPException(422, detail=f"DownloadError yt-dlp: {y.get('stderr', '')[:300]}  (usa 'Carica file' o 'Registra 10s')")
            # se yt-dlp fallisce per altri motivi, proviamo httpx
    if via is None:
        h = await _download_via_httpx(url)
        if not h.get("ok"):
            err = h.get("error")
            if err == "hls_stream":
                raise HTTPException(415, detail={"error": "HLS non supportato in modo diretto",
                                                 "hint": "Registra 10s e carica il file."})
            if err in ("too_large", "too_large_stream"):
                raise HTTPException(413, detail={"error": "File remoto troppo grande"})
            ct = h.get("content_type")
            raise HTTPException(415, detail={"error": f"Unsupported or empty media: {err}", "content_type": ct})

        via = h

    path = via.get("path")
    if not path or not os.path.exists(path):
        raise HTTPException(415, detail={"error": "Download fallito o file assente"})

    try:
        return JSONResponse(_analyze_file(path, source_url=url, platform=via.get("platform")))
    finally:
        pass  # eventuale cleanup differito

# ==== Retro-compat ====
@app.post("/predict")
async def predict(file: UploadFile = File(None),
                  url: Optional[str] = Form(None),
                  link: Optional[str] = Form(None),
                  q: Optional[str] = Form(None)):
    """
    Retro-compat: accetta file oppure url|link|q e reindirizza agli endpoint moderni.
    """
    if file and file.filename:
        return await analyze(file=file)
    the_url = url or link or q
    if the_url:
        return await analyze_url(url=the_url)
    raise HTTPException(415, detail={
        "error": "Unsupported or empty media: no audio/video streams found",
        "ffprobe_found": True,
        "hint": "Se è un URL social, usa analyze-url o abilita yt-dlp; se è un upload, verifica che sia un vero video."
    })
