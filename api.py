# api.py
import os
import io
import shutil
import tempfile
import traceback
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

import subprocess
import json
import httpx
import mimetypes

import app.analyzers.video         as video_an
import app.analyzers.audio         as audio_an
import app.analyzers.meta          as meta_an
import app.analyzers.forensic      as forensic_an
import app.analyzers.fusion        as fusion_an
import app.analyzers.heuristics_v2 as heur_an


# ==== ENV / CONFIG ==========================================================
MAX_UPLOAD_BYTES     = int(os.getenv("MAX_UPLOAD_BYTES",      str(100 * 1024 * 1024)))  # 100MB
RESOLVER_MAX_BYTES   = int(os.getenv("RESOLVER_MAX_BYTES",    str(120 * 1024 * 1024)))
REQUEST_TIMEOUT_S    = int(os.getenv("REQUEST_TIMEOUT_S",     "120"))
ALLOWED_ORIGINS      = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGIN_REGEX = os.getenv("ALLOWED_ORIGIN_REGEX")
USE_YTDLP            = os.getenv("USE_YTDLP", "1") == "1"
RESOLVER_UA          = os.getenv("RESOLVER_UA", "Mozilla/5.0 (AVD)")
YTDLP_OPTS           = os.getenv("YTDLP_OPTS", '{"noplaylist":true,"continuedl":false,"retries":1}')

# ==== APP ===================================================================
app = FastAPI()

# CORS: se non configurato, consenti tutte le origini
allowed_list = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_list if allowed_list else ["*"],
    allow_origin_regex=ALLOWED_ORIGIN_REGEX if ALLOWED_ORIGIN_REGEX else None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== UTILS ================================================================

def _tmpdir(prefix: str = "aiv_") -> str:
    return tempfile.mkdtemp(prefix=prefix)

def _cleanup_dir(path: str):
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

def _fail(status: int, error: str, **extra):
    payload = {"detail": {"error": error}}
    if extra:
        payload["detail"].update(extra)
    return JSONResponse(status_code=status, content=payload)

def _ffprobe_json(path: str) -> Dict[str, Any]:
    cmd = ["ffprobe","-v","error","-print_format","json","-show_format","-show_streams","-show_chapters", path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    try:
        return json.loads(p.stdout or "{}")
    except Exception:
        return {}

def _meta_from_ffprobe(ffj: Dict[str, Any]) -> Dict[str, Any]:
    vstreams = [s for s in ffj.get("streams",[]) if s.get("codec_type")=="video"]
    astreams = [s for s in ffj.get("streams",[]) if s.get("codec_type")=="audio"]
    width = vstreams[0].get("width") if vstreams else None
    height = vstreams[0].get("height") if vstreams else None
    fps = None
    if vstreams:
        r = vstreams[0].get("avg_frame_rate") or vstreams[0].get("r_frame_rate")
        try:
            if r and "/" in r:
                n,d = r.split("/")
                n, d = float(n), float(d or 1)
                fps = n/d if d else None
        except Exception:
            fps = None
    duration = None
    try:
        duration = float(ffj.get("format",{}).get("duration", 0.0))
    except Exception:
        pass
    bit_rate = None
    try:
        bit_rate = int(ffj.get("format",{}).get("bit_rate", 0))
    except Exception:
        pass
    vcodec = vstreams[0].get("codec_name") if vstreams else None
    acodec = astreams[0].get("codec_name") if astreams else None
    fmt    = ffj.get("format",{}).get("format_name")
    return {
        "width": width, "height": height, "fps": fps, "duration": duration,
        "bit_rate": bit_rate, "vcodec": vcodec, "acodec": acodec, "format_name": fmt
    }

def _sniff_is_media(content_type: str, url: str) -> bool:
    if not content_type:
        mt, _ = mimetypes.guess_type(url)
        content_type = mt or ""
    content_type = content_type.lower()
    if "text/html" in content_type: return False
    if "application/vnd.apple.mpegurl" in content_type: return False  # HLS
    if content_type.startswith("video/") or content_type.startswith("audio/"):
        return True
    return False

async def _download_url_to_file(url: str, max_bytes: int) -> str:
    tmpd = _tmpdir("dl_")
    out = os.path.join(tmpd, "media.bin")

    # 1) yt-dlp (se abilitato)
    if USE_YTDLP:
        ytopts = json.loads(YTDLP_OPTS or "{}")
        cmd = ["yt-dlp","-g","--no-warnings"]
        for k,v in ytopts.items():
            if isinstance(v, bool):
                if v: cmd.append(f"--{k}")
            else:
                cmd += [f"--{k}", str(v)]
        cmd.append(url)
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if p.returncode == 0 and p.stdout.strip():
            direct = p.stdout.strip().splitlines()[0]
            url = direct  # rimpiazza con stream diretto

    # 2) httpx con UA e limite dimensione
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S, headers={"User-Agent": RESOLVER_UA}) as client:
        r = await client.get(url, follow_redirects=True)
        ct = r.headers.get("content-type","")
        if not _sniff_is_media(ct, url):
            # rate limit / login wall → 422 con hint
            if "text/html" in (ct or "").lower():
                raise HTTPException(status_code=422, detail="Login required, paywall o pagina HTML (usa upload o screen recording).")
            raise HTTPException(status_code=415, detail="MIME non supportato o non multimediale.")
        data = r.content
        if len(data) == 0:
            raise HTTPException(status_code=415, detail="File vuoto o non ricevuto")
        if len(data) > max_bytes:
            raise HTTPException(status_code=413, detail=f"File troppo grande (> {max_bytes} bytes)")
        with open(out, "wb") as f:
            f.write(data)
    return out

def _analyze_file(path: str) -> Dict[str, Any]:
    ffj = _ffprobe_json(path)
    meta_base = _meta_from_ffprobe(ffj)

    # meta esteso + forensic (sempre con chiave 'present' booleana)
    meta_ext = {"meta": meta_base}
    try:
        meta_device = meta_an.detect_device(path)
    except Exception:
        meta_device = {}
    try:
        meta_c2pa = meta_an.detect_c2pa(path)
    except Exception:
        meta_c2pa = {"present": False}

    forensic = forensic_an.analyze(path)
    meta_ext.update(meta_device)
    meta_ext["forensic"] = {"c2pa": {"present": bool(meta_c2pa.get("present", False))}}
    meta_ext["ffprobe_raw"] = bool(ffj)

    # analysis
    vstats = video_an.analyze(path, max_seconds=30)
    astats = audio_an.analyze(path, target_sr=16000)

    # hints (facoltativi)
    hints = heur_an.compute_hints(vstats, astats, meta_ext)

    fused = fusion_an.fuse(vstats, astats, {"meta": meta_base, "forensic": {"c2pa": meta_c2pa}})
    # aggrega e restituisce
    return {
        "ok": True,
        "meta": meta_base,
        "forensic": {"c2pa": {"present": bool(meta_c2pa.get("present", False))}},
        "video": vstats,
        "audio": astats,
        "hints": hints,
        **fused
    }

# ==== ROUTES ===============================================================

@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

# Endpoint diagnostico CORS: accetta GET/POST/OPTIONS per evitare 405
@app.api_route("/cors-test", methods=["GET", "POST", "OPTIONS"])
async def cors_test(request: Request):
    return JSONResponse({
        "ok": True,
        "method": request.method,
        "origin": request.headers.get("origin"),
        "access_control_request_method": request.headers.get("access-control-request-method"),
        "access_control_request_headers": request.headers.get("access-control-request-headers"),
    })

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file or not file.filename:
        return _fail(415, "File vuoto o non ricevuto")
    # limita dimensione
    data = await file.read()
    if len(data) == 0:
        return _fail(415, "File vuoto o non ricevuto")
    if len(data) > MAX_UPLOAD_BYTES:
        return _fail(413, f"File troppo grande (> {MAX_UPLOAD_BYTES} bytes)")
    tmpd = _tmpdir("up_")
    path = os.path.join(tmpd, file.filename)
    with open(path, "wb") as f:
        f.write(data)
    try:
        out = _analyze_file(path)
        return JSONResponse(out)
    except HTTPException as e:
        return _fail(e.status_code, str(e.detail))
    except Exception as e:
        return _fail(500, "Errore interno", traceback=traceback.format_exc())
    finally:
        _cleanup_dir(tmpd)

@app.post("/analyze-url")
async def analyze_url(url: str = Form(...)):
    url = (url or "").strip()
    if not url:
        return _fail(422, "URL mancante")
    try:
        path = await _download_url_to_file(url, RESOLVER_MAX_BYTES)
    except HTTPException as e:
        return _fail(e.status_code, str(e.detail))
    except Exception as e:
        # yt-dlp: login required / rate limit message?
        msg = str(e)
        if "Login required" in msg or "cookies" in msg.lower():
            return _fail(422, "Login richiesto o rate-limit: usa upload o screen-recording")
        return _fail(422, f"DownloadError: {msg[:200]}")
    try:
        out = _analyze_file(path)
        return JSONResponse(out)
    except HTTPException as e:
        return _fail(e.status_code, str(e.detail))
    except Exception as e:
        return _fail(500, "Errore interno", traceback=traceback.format_exc())
    finally:
        _cleanup_dir(os.path.dirname(path))

@app.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    url:  Optional[str] = Form(None),
    link: Optional[str] = Form(None),
    q:    Optional[str] = Form(None),
):
    """
    Retro-compatibile:
      - se arriva un file → /analyze
      - se arriva un URL → /analyze-url
    """
    if file is not None:
        return await analyze(file=file)
    the_url = (url or link or q or "").strip()
    if the_url:
        return await analyze_url(url=the_url)
    return _fail(415, "File vuoto o non ricevuto")

# Preflight universale (copre ogni path OPTIONS)
@app.options("/{path:path}")
async def preflight(path: str):
    return PlainTextResponse("OK", status_code=200)
