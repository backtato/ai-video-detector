# api.py
import os
import io
import shutil
import tempfile
import traceback
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

import subprocess
import json
import httpx
import mimetypes
import re

# ==== Config da ENV (con default sicuri) ====
MAX_UPLOAD_BYTES    = int(os.getenv("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))  # 100 MB
RESOLVER_MAX_BYTES  = int(os.getenv("RESOLVER_MAX_BYTES", str(120 * 1024 * 1024)))
REQUEST_TIMEOUT_S   = int(os.getenv("REQUEST_TIMEOUT_S", "45"))
ALLOWED_ORIGINS_ENV = os.getenv("ALLOWED_ORIGINS", "*")
USE_YTDLP           = os.getenv("USE_YTDLP", "0") in ("1", "true", "True", "yes")
YTDLP_OPTS          = os.getenv("YTDLP_OPTS", "")  # JSON str (es: {"cookies-from-browser":"chrome"})
BLOCK_HLS           = os.getenv("BLOCK_HLS", "1") in ("1","true","True","yes")
HEALTH_AUTHOR       = os.getenv("HEALTH_AUTHOR", "Backtato")
VERSION             = os.getenv("VERSION", "1.2.0")

# ==== App ====
app = FastAPI(title="AI-Video Detector", version=VERSION)

# CORS: se ALLOWED_ORIGINS non è "*", splitta per virgola; altrimenti apri a tutti
if ALLOWED_ORIGINS_ENV.strip() == "*":
    cors_origins = ["*"]
else:
    cors_origins = [o.strip() for o in ALLOWED_ORIGINS_ENV.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Import locali (dopo init app) ====
from app.analyzers import audio as audio_an
from app.analyzers import video as video_an
from app.analyzers import fusion as fusion_an
from app.analyzers import meta as meta_an
from app.analyzers import forensic as forensic_an

# ==== Utils ====
def _tmpdir(prefix="tmp_") -> str:
    return tempfile.mkdtemp(prefix=prefix)

def _fail(status: int, msg: str, extra: Dict[str, Any] = None) -> JSONResponse:
    payload = {"detail": {"error": msg}}
    if extra:
        payload["detail"].update(extra)
    return JSONResponse(payload, status_code=status)

def _read_n_bytes(stream: io.BufferedReader, n: int) -> bytes:
    return stream.read(n)

def _is_hls_url(u: str) -> bool:
    return ".m3u8" in u.lower() or re.search(r"format=(?:hls|m3u8)", u, re.I) is not None

def _ffprobe_ok() -> bool:
    try:
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return True
    except Exception:
        return False

def _exiftool_ok() -> bool:
    try:
        subprocess.run(["exiftool", "-ver"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return True
    except Exception:
        return False

async def _download_with_httpx(url: str, out_path: str) -> Dict[str, Any]:
    """
    Download binario via httpx con sniff del Content-Type e guardie dimensione.
    """
    timeout = httpx.Timeout(REQUEST_TIMEOUT_S)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        r = await client.get(url)
        if r.status_code >= 400:
            return {"ok": False, "status": r.status_code, "why": f"GET failed {r.status_code}"}

        ctype = r.headers.get("Content-Type", "").lower()
        if "text/html" in ctype:
            return {"ok": False, "status": 415, "why": "URL restituisce HTML (probabile login-wall o pagina web)"}

        size = int(r.headers.get("Content-Length", "0") or "0")
        if size and size > RESOLVER_MAX_BYTES:
            return {"ok": False, "status": 413, "why": f"File troppo grande ({size} > {RESOLVER_MAX_BYTES})"}

        # streaming su disco
        total = 0
        with open(out_path, "wb") as f:
            async for chunk in r.aiter_bytes():
                if not chunk:
                    continue
                total += len(chunk)
                if total > RESOLVER_MAX_BYTES:
                    return {"ok": False, "status": 413, "why": "Superata dimensione massima durante il download"}
                f.write(chunk)

    return {"ok": True, "status": 200, "ctype": ctype or "", "path": out_path}

def _yt_dlp_direct_url(url: str) -> Dict[str, Any]:
    """
    Prova ad ottenere un URL diretto allo stream con yt-dlp -g.
    Richiede USE_YTDLP=1. Supporta opzioni JSON in YTDLP_OPTS.
    """
    ytopts = {}
    try:
        if YTDLP_OPTS:
            ytopts = json.loads(YTDLP_OPTS)
    except Exception:
        pass

    cmd = ["yt-dlp", "-g", "--no-warnings"]
    for k, v in ytopts.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    cmd.append(url)

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if p.returncode != 0:
        return {"ok": False, "status": 422, "why": f"yt-dlp error: {p.stderr.strip()[:300]}"}

    direct = p.stdout.strip().splitlines()
    if not direct:
        return {"ok": False, "status": 422, "why": "yt-dlp non ha fornito URL diretto"}

    # scegli il primo non-HLS
    for u in direct:
        if not _is_hls_url(u):
            return {"ok": True, "status": 200, "direct": u}
    # se solo HLS e blocchiamo HLS:
    if BLOCK_HLS:
        return {"ok": False, "status": 415, "why": "Solo HLS disponibili; abilita HLS o usa upload/registrazione"}
    return {"ok": True, "status": 200, "direct": direct[0]}

def _label_from_score(score: float) -> str:
    if score <= 0.35:
        return "real"
    if score >= 0.72:
        return "ai"
    return "uncertain"

def _confidence_from_payload(p: Dict[str, Any]) -> int:
    # semplice euristica: clamp 10..99, usa spread dei punteggi + qualità
    try:
        conf = p.get("result", {}).get("confidence", None)
        if conf is not None:
            return int(conf)
    except Exception:
        pass
    return 70  # fallback coerente con tua UI

# ==== HEALTH ====
@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "ffprobe": _ffprobe_ok(),
        "exiftool": _exiftool_ok(),
        "version": VERSION,
        "author": HEALTH_AUTHOR
    }

# ==== CORS TEST ====
@app.get("/cors-test")
@app.post("/cors-test")
async def cors_test(request: Request):
    return {"ok": True, "method": request.method, "origin": request.headers.get("Origin")}

# ==== ANALYZE (UPLOAD) ====
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        # 1) file guardie
        if not file or not file.filename:
            return _fail(415, "File vuoto o non ricevuto")

        # type sniff
        mime = file.content_type or mimetypes.guess_type(file.filename)[0] or ""
        if "video" not in mime and "quicktime" not in mime and not file.filename.lower().endswith((".mp4",".mov",".m4v",".mkv",".webm",".avi",".3gp",".3g2",".ts",".mts",".wmv",".flv")):
            # non blocchiamo troppo: può capitare che i browser mandino octet-stream
            pass

        # 2) salva su disco
        tmpd = _tmpdir("up_")
        path = os.path.join(tmpd, file.filename.replace("/", "_"))
        size = 0
        with open(path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    try: os.remove(path)
                    except: pass
                    return _fail(413, "File troppo grande", {"max_bytes": MAX_UPLOAD_BYTES})
                f.write(chunk)

        # 3) analisi
        return _analyze_file(path)

    except Exception as e:
        return _fail(500, "Errore interno (analyze)", {"trace": traceback.format_exc()[:1200]})

# ==== ANALYZE-URL ====
@app.post("/analyze-url")
async def analyze_url(url: str = Form(...)):
    try:
        url = (url or "").strip()
        if not url:
            return _fail(415, "URL mancante")

        if BLOCK_HLS and _is_hls_url(url):
            return _fail(415, "Formati HLS non supportati in questa istanza")

        tmpd = _tmpdir("dl_")
        out = os.path.join(tmpd, "media.bin")

        # 1) tenta yt-dlp se abilitato
        if USE_YTDLP:
            r = _yt_dlp_direct_url(url)
            if r.get("ok"):
                url = r["direct"]
            else:
                # rate-limit / login-wall → messaggio guidato
                why = r.get("why", "Impossibile risolvere l'URL con yt-dlp")
                return _fail(r.get("status", 422), why, {"hint": "Usa 'Carica file' o 'Registra 10s' se il link è protetto"})

        # 2) download diretto
        got = await _download_with_httpx(url, out)
        if not got.get("ok"):
            return _fail(got.get("status", 415), got.get("why", "Download fallito"))

        ctype = got.get("ctype", "")
        if ("text/html" in ctype) or (not ctype and not out.lower().endswith((".mp4",".mov",".m4v",".mkv",".webm",".avi",".3gp",".3g2",".ts",".mts",".wmv",".flv"))):
            return _fail(415, "Non sembra un file video diretto", {"content_type": ctype, "hint": "Se è un social protetto, usa upload/registrazione o abilita yt-dlp con cookie"})

        # 3) analisi
        return _analyze_file(out, source_url=url)

    except Exception as e:
        return _fail(500, "Errore interno (analyze-url)", {"trace": traceback.format_exc()[:1200]})

# ==== PREDICT (retro-compatibile) ====
@app.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    url: Optional[str]       = Form(None),
    link: Optional[str]      = Form(None),
    q: Optional[str]         = Form(None),
):
    """
    Smista:
      - se arriva un file → /analyze
      - se arriva un URL → /analyze-url
    """
    if file is not None:
        return await analyze(file=file)
    the_url = (url or link or q or "").strip()
    if the_url:
        return await analyze_url(url=the_url)
    return _fail(415, "File vuoto o non ricevuto")

# ==== Preflight universale (copre ogni path OPTIONS) ====
@app.options("/{path:path}")
async def preflight(path: str):
    return PlainTextResponse("OK", status_code=200)

# ==== Core analysis ====
def _analyze_file(path: str, source_url: Optional[str] = None) -> JSONResponse:
    # forense leggera + device + c2pa (fault-tolerant)
    forensic = {}
    try:
        forensic = forensic_an.analyze(path)
    except Exception:
        forensic = {"c2pa": {"present": False}}

    meta_device = {}
    try:
        meta_device = meta_an.detect_device(path)
    except Exception:
        meta_device = {"device": {"vendor": None, "model": None, "os": "Unknown"}}

    c2pa = {}
    try:
        c2pa = meta_an.detect_c2pa(path)
    except Exception:
        c2pa = {"present": False, "note": "c2pa non determinato"}

    # video / audio
    vstats = {}
    astats = {}
    meta = {}
    try:
        vstats = video_an.analyze(path) or {}
    except Exception:
        vstats = {}
    try:
        astats = audio_an.analyze(path) or {}
    except Exception:
        astats = {}
    try:
        # riduci metadati essenziali per UI
        meta = {
            "width":  int(vstats.get("width") or 0),
            "height": int(vstats.get("height") or 0),
            "fps":    float(vstats.get("src_fps") or 0.0),
            "duration": float(vstats.get("duration") or 0.0),
            "bit_rate": int(vstats.get("bit_rate") or 0),
            "vcodec":  vstats.get("vcodec"),
            "acodec":  vstats.get("acodec"),
            "format_name": vstats.get("format_name"),
            "source_url": source_url,
            "resolved_url": source_url,
            "forensic": {"c2pa": {"present": bool(c2pa.get("present"))}},
            "device": meta_device.get("device", {}),
        }
    except Exception:
        meta = {"source_url": source_url, "resolved_url": source_url}

    # fusione conservativa
    fused = {}
    try:
        fused = fusion_an.fuse(video_stats=vstats, audio_stats=astats, meta=meta, c2pa=c2pa)
    except Exception:
        # fallback neutro
        fused = {
            "result": {"label": "uncertain", "ai_score": 0.5, "confidence": 60, "reason": "analisi parziale"},
            "timeline_binned": [],
            "peaks": [],
            "hints": {}
        }

    payload = {
        "ok": True,
        "meta": meta,
        "forensic": {"c2pa": {"present": bool(c2pa.get("present"))}},
        "video": vstats or {},
        "audio": astats or {},
        "hints": fused.get("hints", {}),
        "result": fused.get("result", {}),
        "timeline_binned": fused.get("timeline_binned", []),
        "peaks": fused.get("peaks", []),
    }
    # UI-friendly clamp
    payload["result"]["label"] = payload["result"].get("label") or _label_from_score(payload["result"].get("ai_score", 0.5))
    payload["result"]["confidence"] = _confidence_from_payload(payload)

    return JSONResponse(payload, status_code=200)
