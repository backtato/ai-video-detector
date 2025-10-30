import os
import io
import json
import tempfile
import subprocess
import asyncio
import logging
import traceback
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from app.analyzers import audio as audio_an
from app.analyzers import video as video_an
from app.analyzers import fusion as fusion_an
from app.analyzers import heuristics_v2 as hx
from app.analyzers import meta as meta_an

VERSION = os.getenv("VERSION", "1.2.2")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
RESOLVER_MAX_BYTES = int(os.getenv("RESOLVER_MAX_BYTES", str(120 * 1024 * 1024)))
REQUEST_TIMEOUT_S = int(os.getenv("REQUEST_TIMEOUT_S", "180"))
USE_YTDLP = os.getenv("USE_YTDLP", "1") == "1"
DEBUG = os.getenv("DEBUG", "0") == "1"

# ----- App & CORS -----
app = FastAPI()
allow_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins if allow_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Utilities -----
def _which(cmd: str) -> Optional[str]:
    try:
        return subprocess.check_output(["bash","-lc", f"command -v {cmd}"], text=True).strip() or None
    except Exception:
        return None

def _run_ffprobe(path: str) -> Dict[str, Any]:
    try:
        cmd = [
            "ffprobe","-v","error","-show_entries",
            "format=bit_rate,duration,format_name:stream=codec_name,codec_type,width,height,r_frame_rate",
            "-of","json", path
        ]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=30)
        return json.loads(out)
    except Exception:
        return {}

def _probe_basic_meta(path: str) -> Dict[str, Any]:
    info = _run_ffprobe(path)
    width = height = fps = 0.0
    vcodec = acodec = None
    duration = 0.0
    if info.get("streams"):
        for s in info["streams"]:
            if s.get("codec_type") == "video" and not width:
                width = float(s.get("width") or 0)
                height = float(s.get("height") or 0)
                r = s.get("r_frame_rate") or "0/1"
                try:
                    num, den = r.split("/")
                    fps = float(num) / max(1.0, float(den))
                except Exception:
                    fps = 0.0
                vcodec = s.get("codec_name")
            elif s.get("codec_type") == "audio" and not acodec:
                acodec = s.get("codec_name")
    bit_rate = 0
    fmt = None
    if info.get("format"):
        bit_rate = int(float(info["format"].get("bit_rate") or 0))
        fmt = info["format"].get("format_name")
        try:
            duration = float(info["format"].get("duration") or 0.0)
        except Exception:
            duration = 0.0
    return {
        "width": int(width), "height": int(height), "fps": fps, "duration": duration,
        "bit_rate": bit_rate, "vcodec": vcodec, "acodec": acodec, "format_name": fmt
    }

def _save_upload_to_tmp(upload: UploadFile, max_bytes: int) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload.filename or "")[1] or ".bin")
    size = 0
    try:
        with tmp as f:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, detail={"error":"File troppo grande","limit_bytes":max_bytes})
                f.write(chunk)
        return tmp.name
    except Exception:
        try: os.unlink(tmp.name)
        except Exception: pass
        raise

def _ready_probe() -> Dict[str, Any]:
    return {
        "ffprobe": bool(_which("ffprobe")),
        "exiftool": bool(_which("exiftool")),
        "version": VERSION,
        "author": "Backtato",
    }

async def _analyze_path(path: str, source_url: Optional[str]=None, resolved_url: Optional[str]=None) -> Dict[str, Any]:
    meta = _probe_basic_meta(path)
    hints = hx.compute_hints(meta, path)
    # Run audio/video in threads to avoid blocking
    audio_task = asyncio.to_thread(audio_an.analyze, path, meta)
    video_task = asyncio.to_thread(video_an.analyze, path, meta)
    audio = await asyncio.wait_for(audio_task, timeout=REQUEST_TIMEOUT_S)
    video = await asyncio.wait_for(video_task, timeout=REQUEST_TIMEOUT_S)
    fused = fusion_an.fuse(audio, video, hints)
    out = {
        "ok": True,
        "meta": {
            **meta,
            "source_url": source_url,
            "resolved_url": resolved_url
        },
        "hints": hints,
        "video": video,
        "audio": audio,
        "result": fused["result"],
        "timeline_binned": fused["timeline_binned"],
        "peaks": fused["peaks"],
    }
    # Forensic/meta extras
    try:
        forensic = meta_an.forensic_summary(path)
        if forensic:
            out["forensic"] = forensic
    except Exception:
        if DEBUG:
            out["forensic_error"] = traceback.format_exc()
    return out

def _yt_dlp_download(url: str, max_bytes: int) -> Dict[str, Any]:
    """Download a media file using yt-dlp without cookies."""
    if not USE_YTDLP:
        raise HTTPException(422, detail={"error":"yt-dlp disabilitato","hint":"Abilita USE_YTDLP=1"})
    import yt_dlp
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()
    base_opts = {
        "outtmpl": tmp.name,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 1,
        "user_agent": os.getenv("RESOLVER_UA","Mozilla/5.0 (AVD/1.2)"),
        "http_headers": {"User-Agent": os.getenv("RESOLVER_UA","Mozilla/5.0 (AVD/1.2)")},
        "format": "bv*+ba/best",
        "max_filesize": max_bytes,
        "nocheckcertificate": True,
        "geo_bypass": True,
        "overwrites": True,
    }
    try:
        with yt_dlp.YoutubeDL(base_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return {"path": tmp.name, "resolved_url": info.get("url") or info.get("webpage_url") or url}
    except yt_dlp.utils.DownloadError as e:
        try: os.unlink(tmp.name)
        except Exception: pass
        msg = str(e).lower()
        if "login" in msg or "private" in msg or "cookies" in msg:
            raise HTTPException(415, detail={"error":"Contenuto protetto da login / cookies","hint":"Usa 'Carica file' o 'Registra 10s'."})
        if "unsupported url" in msg:
            raise HTTPException(415, detail={"error":"URL non supportato","hint":"Prova con un link diretto o carica il file."})
        if "filesize" in msg or "too large" in msg:
            raise HTTPException(413, detail={"error":"File troppo grande dal provider","limit_bytes": max_bytes})
        raise HTTPException(415, detail={"error":"Errore di download","hint":"Rate limit o blocco. Riprova o carica il file."})
    except Exception as e:
        try: os.unlink(tmp.name)
        except Exception: pass
        raise HTTPException(415, detail={"error":"Impossibile scaricare il video","exception":str(e)})

# ----- Routes -----
@app.get("/", response_class=JSONResponse)
def root():
    return {"ok": True, "service": "ai-video-detector", "version": VERSION}

@app.get("/healthz", response_class=JSONResponse)
def healthz():
    # as-light-as-possible
    return {"ok": True, "version": VERSION}

@app.get("/readyz", response_class=JSONResponse)
def readyz():
    return {"ok": True, **_ready_probe()}

@app.options("/{path:path}")
async def options_preflight(path: str):
    from fastapi.responses import Response
    return Response(status_code=204)

@app.post("/cors-test", response_class=JSONResponse)
async def cors_test(request: Request):
    body = await request.body()
    return {"ok": True, "echo": body.decode("utf-8", "ignore")}

@app.post("/analyze", response_class=JSONResponse)
async def analyze(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(415, detail={"error":"File vuoto o non ricevuto"})
    # Save to tmp (chunked)
    path = _save_upload_to_tmp(file, MAX_UPLOAD_BYTES)
    try:
        result = await asyncio.wait_for(_analyze_path(path), timeout=REQUEST_TIMEOUT_S)
        return JSONResponse(result)
    finally:
        try: os.unlink(path)
        except Exception: pass

@app.post("/predict", response_class=JSONResponse)
async def predict(file: UploadFile = File(None), url: str = Form(None)):
    if file is not None:
        return await analyze(file=file)
    if url:
        return await analyze_url(url=url)
    raise HTTPException(422, detail={"error":"Nessun input","hint":"Invia 'file' oppure 'url'."})

@app.post("/analyze-url", response_class=JSONResponse)
async def analyze_url(url: str = Form(...)):
    if not url:
        raise HTTPException(422, detail={"error":"URL mancante"})
    dl = _yt_dlp_download(url, RESOLVER_MAX_BYTES)
    path = dl["path"]
    try:
        result = await asyncio.wait_for(_analyze_path(path, source_url=url, resolved_url=dl.get("resolved_url")), timeout=REQUEST_TIMEOUT_S)
        return JSONResponse(result)
    finally:
        try: os.unlink(path)
        except Exception: pass

# ----- Error handling -----
@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    if DEBUG:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "detail":{
                "error": str(exc),
                "exception": exc.__class__.__name__,
                "traceback": traceback.format_exc(),
            }},
        )
    return JSONResponse(status_code=500, content={"ok": False, "detail":{"error":"Internal server error"}})