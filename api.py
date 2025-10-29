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

# CORS
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

# ==== Import locali ====
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

    for u in direct:
        if not _is_hls_url(u):
            return {"ok": True, "status": 200, "direct": u}
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
    try:
        conf = p.get("result", {}).get("confidence", None)
        if conf is not None:
            return int(conf)
    except Exception:
        pass
    return 70  # fallback

# === Bitrate/BPP helpers ===
def _infer_bitrate(meta: dict, file_size_bytes: int) -> int:
    br = int(meta.get("bit_rate") or 0)
    dur = float(meta.get("duration") or 0.0)
    if br <= 0 and file_size_bytes and dur > 0:
        br = int((file_size_bytes * 8) / dur)  # bit/s
    return br

def _compute_bpp(meta: dict, br_bits_per_s: int) -> float:
    w = int(meta.get("width") or 0)
    h = int(meta.get("height") or 0)
    fps = float(meta.get("fps") or 0.0)
    if br_bits_per_s <= 0 or w <= 0 or h <= 0 or fps <= 0:
        return 0.0
    # bits per pixel per frame ~ bitrate / (fps * pixels_per_frame)
    return float(br_bits_per_s) / (fps * (w * h))

def _clamp_timeline(timeline: List[Dict[str, Any]], duration_s: float) -> List[Dict[str, Any]]:
    """Clamp generico di una timeline [start,end) alla durata indicata."""
    if not timeline or duration_s <= 0:
        return timeline or []
    out: List[Dict[str, Any]] = []
    for b in timeline:
        s = float(b.get("start", 0))
        e = float(b.get("end", s + 1))
        if s >= duration_s:
            break
        bb = dict(b)
        bb["end"] = min(e, duration_s)
        out.append(bb)
    return out

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
        if not file or not file.filename:
            return _fail(415, "File vuoto o non ricevuto")

        # salvataggio su disco con limite dimensione
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

        return _analyze_file(path)

    except Exception:
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

        if USE_YTDLP:
            r = _yt_dlp_direct_url(url)
            if r.get("ok"):
                url = r["direct"]
            else:
                why = r.get("why", "Impossibile risolvere l'URL con yt-dlp")
                return _fail(r.get("status", 422), why, {"hint": "Usa 'Carica file' o 'Registra 10s' se il link è protetto"})

        got = await _download_with_httpx(url, out)
        if not got.get("ok"):
            return _fail(got.get("status", 415), got.get("why", "Download fallito"))

        ctype = got.get("ctype", "")
        if ("text/html" in ctype) or (not ctype and not out.lower().endswith((".mp4",".mov",".m4v",".mkv",".webm",".avi",".3gp",".3g2",".ts",".mts",".wmv",".flv"))):
            return _fail(415, "Non sembra un file video diretto", {"content_type": ctype, "hint": "Se è un social protetto, usa upload/registrazione o abilita yt-dlp con cookie"})

        return _analyze_file(out, source_url=url)

    except Exception:
        return _fail(500, "Errore interno (analyze-url)", {"trace": traceback.format_exc()[:1200]})

# ==== PREDICT (retro-compatibile) ====
@app.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    url: Optional[str]       = Form(None),
    link: Optional[str]      = Form(None),
    q: Optional[str]         = Form(None),
):
    if file is not None:
        return await analyze(file=file)
    the_url = (url or link or q or "").strip()
    if the_url:
        return await analyze_url(url=the_url)
    return _fail(415, "File vuoto o non ricevuto")

# ==== Preflight universale ====
@app.options("/{path:path}")
async def preflight(path: str):
    return PlainTextResponse("OK", status_code=200)

# ==== Core analysis ====
def _analyze_file(path: str, source_url: Optional[str] = None) -> JSONResponse:
    # forense + device + c2pa (tollerante agli errori)
    try:
        forensic = forensic_an.analyze(path)
    except Exception:
        forensic = {"c2pa": {"present": False}}

    try:
        meta_device = meta_an.detect_device(path)
    except Exception:
        meta_device = {"device": {"vendor": None, "model": None, "os": "Unknown"}}

    try:
        c2pa = meta_an.detect_c2pa(path)
    except Exception:
        c2pa = {"present": False, "note": "c2pa non determinato"}

    # video / audio
    try:
        vstats = video_an.analyze(path) or {}
    except Exception:
        vstats = {}
    try:
        astats = audio_an.analyze(path) or {}
    except Exception:
        astats = {}

    # ⬅️ Clamp finale timeline audio alla durata video (evita bucket 16→16.36 ecc.)
    try:
        vdur = float(vstats.get("duration") or 0.0)
        if vdur > 0 and astats.get("timeline"):
            astats["timeline"] = _clamp_timeline(astats["timeline"], vdur)
    except Exception:
        pass

    # meta essenziali per UI
    try:
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

    # ==== Hints (bpp/compressione, signal) + backfill bitrate in meta ====
    try:
        file_size_bytes = os.path.getsize(path)
    except Exception:
        file_size_bytes = 0

    bitrate = _infer_bitrate(meta, file_size_bytes)
    bpp = _compute_bpp(meta, bitrate)

    compression = "normal"
    if bpp < 0.06:
        compression = "heavy"
    elif bpp < 0.10:
        compression = "moderate"

    # backfill in meta se mancava
    if int(meta.get("bit_rate") or 0) <= 0 and bitrate > 0:
        meta["bit_rate"] = bitrate

    vsummary = vstats.get("summary") or {}
    motion_used = float(vsummary.get("motion_avg") or 0.0)
    flow_used = float(vsummary.get("optflow_mag_avg") or 0.0)
    video_has_signal = bool(vstats.get("timeline_ai"))

    hints = {
        "bpp": round(bpp, 5),
        "compression": compression,
        "video_has_signal": video_has_signal,
        "flow_used": flow_used,
        "motion_used": motion_used,
        "w": meta.get("width") or 0,
        "h": meta.get("height") or 0,
        "fps": meta.get("fps") or 0.0,
        "br": bitrate or 0
    }

    # ==== Fusione conservativa (per-secondo reale) ====
    try:
        fused = fusion_an.fuse(video=vstats, audio=astats, hints=hints)
        result = fused.get("result", {})
        timeline_binned = fused.get("timeline_binned", [])
        peaks = fused.get("peaks", [])
        hints_out = fused.get("hints", hints)
    except Exception:
        result = {"label": "uncertain", "ai_score": 0.5, "confidence": 60, "reason": "analisi parziale"}
        timeline_binned = []
        peaks = []
        hints_out = hints

    payload = {
        "ok": True,
        "meta": meta,
        "forensic": {"c2pa": {"present": bool(c2pa.get("present"))}},
        "video": vstats or {},
        "audio": astats or {},
        "hints": hints_out,
        "result": result,
        "timeline_binned": timeline_binned,
        "peaks": peaks,
    }

    payload["result"]["label"] = payload["result"].get("label") or _label_from_score(payload["result"].get("ai_score", 0.5))
    payload["result"]["confidence"] = _confidence_from_payload(payload)

    return JSONResponse(payload, status_code=200)
