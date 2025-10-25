# api.py
import os, io, json, shutil, tempfile, traceback, hashlib
from typing import Optional, Tuple
from urllib.parse import urlparse

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from diskcache import Cache
from yt_dlp import YoutubeDL

from app.analyzers.meta import extract_metadata, detect_c2pa, detect_device_fingerprint
from app.analyzers.video import analyze_video
from app.analyzers.audio import analyze_audio
from app.fusion import fuse_scores, finalize_with_timeline, _bin_timeline, build_hints

APP_NAME = "AI-Video Detector"
DETECTOR_VERSION = os.getenv("DETECTOR_VERSION", "1.2.0")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50*1024*1024)))
RESOLVER_MAX_BYTES = int(os.getenv("RESOLVER_MAX_BYTES", str(120*1024*1024)))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
USE_YTDLP = os.getenv("USE_YTDLP", "1") not in ("0","false","False")
RESOLVER_UA = os.getenv("RESOLVER_UA", "Mozilla/5.0 (compatible; AI-Video/1.0)")
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/aivideo-cache")

os.makedirs(CACHE_DIR, exist_ok=True)
cache = Cache(CACHE_DIR)

DEFAULT_YTDLP_OPTS = {
    "noplaylist": True,
    "continuedl": False,
    "retries": 1,
    "quiet": True,
}

ALLOW_DOMAINS = {
    "youtube.com","www.youtube.com","youtu.be",
    "vimeo.com","www.vimeo.com",
    "instagram.com","www.instagram.com",
    "tiktok.com","www.tiktok.com",
    "facebook.com","www.facebook.com","fb.watch",
    "x.com","www.x.com","twitter.com","www.twitter.com",
}
ALLOW_EXTS = (".mp4",".mov",".m4v",".webm",".mpg",".mpeg",".avi",".mkv",".m4a")

app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _raise_422(msg: str):
    raise HTTPException(status_code=422, detail=msg)

def _is_allowed_url(url: str) -> bool:
    try:
        u = urlparse(url)
        return u.scheme in ("http","https") and any((u.netloc or "").lower().endswith(d) for d in ALLOW_DOMAINS)
    except: return False

def _file_hash(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def _disk_cached(key: str) -> Optional[dict]:
    try: return cache.get(key)
    except: return None

def _disk_store(key: str, value: dict, expire: int = 3600):
    try: cache.set(key, value, expire=expire)
    except: pass

def _ytdlp_resolve_and_download(url: str) -> Tuple[str, str]:
    opts = dict(DEFAULT_YTDLP_OPTS)
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
            if not info: _raise_422("Impossibile estrarre il video (yt-dlp).")
            fpath = None
            if "requested_downloads" in info and info["requested_downloads"]:
                fpath = info["requested_downloads"][0].get("filepath")
            if not fpath:
                title = info.get("title","video"); ext = info.get("ext","mp4")
                fpath = os.path.join(tmpdir, f"{title}.{ext}")
            if not os.path.exists(fpath):
                cand = [os.path.join(tmpdir,p) for p in os.listdir(tmpdir) if os.path.isfile(os.path.join(tmpdir,p))]
                if not cand: _raise_422("Download non riuscito (yt-dlp).")
                fpath = max(cand, key=lambda p: os.path.getsize(p))
            if os.path.getsize(fpath) > RESOLVER_MAX_BYTES:
                try: shutil.rmtree(tmpdir)
                except: pass
                _raise_422("File troppo grande dal provider. Usa 'Registra 10s' o 'Carica file'.")
            return fpath, info.get("webpage_url") or url
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ["login","cookies","protected","rate-limit","authentication"]):
            _raise_422("Protected content o login richiesto. Suggerimento: usa 'Carica file' o 'Registra 10s'.")
        _raise_422(f"Impossibile scaricare: {e}")

def _analyze_path(path: str, source_url: Optional[str]=None, resolved_url: Optional[str]=None) -> dict:
    meta = extract_metadata(path)
    if source_url is not None: meta["source_url"] = source_url
    if resolved_url is not None: meta["resolved_url"] = resolved_url

    c2pa = detect_c2pa(path)
    dev_fp = detect_device_fingerprint(meta)

    v_res = analyze_video(path)
    a_res = analyze_audio(path)

    hints = build_hints(meta, v_res.get("flags_video"), c2pa_present=c2pa.get("present",False), dev_fp=dev_fp)
    fusion_core = fuse_scores(
        frame=v_res["scores"].get("frame_mean",0.5),
        audio=a_res["scores"].get("audio_mean",0.5),
        hints=hints
    )

    timeline = (v_res.get("timeline") or []) + (a_res.get("timeline") or [])
    duration = meta.get("duration") or 0.0
    base_bins = _bin_timeline(timeline, duration=duration or 0.0, bin_sec=1.0, mode="max")
    fusion, peaks = finalize_with_timeline(fusion_core, base_bins)

    tips = []
    if fusion["label"] == "uncertain": tips.append("Esito incerto: prova un video più lungo o di qualità migliore.")
    if hints.get("no_c2pa"): tips.append("Nessuna firma C2PA rilevata (non prova di falsità).")
    if "long_pauses" in (a_res.get("flags_audio") or []): tips.append("Audio con molte pause o a bassa energia.")
    if "low_motion" in (v_res.get("flags_video") or []): tips.append("Movimento basso/prolungato: analisi più difficile.")

    return {
        "ok": True,
        "meta": meta,
        "forensic": {"c2pa": c2pa, "device_fingerprint": dev_fp},
        "scores": {
            "frame_mean": v_res["scores"].get("frame_mean"),
            "frame_std": v_res["scores"].get("frame_std"),
            "audio_mean": a_res["scores"].get("audio_mean"),
        },
        "video": v_res.get("video"),
        "flags_video": v_res.get("flags_video"),
        "flags_audio": a_res.get("flags_audio"),
        "fusion": {
            "ai_score": fusion["ai_score"],
            "label": fusion["label"],
            "confidence": fusion["confidence"],
        },
        "timeline": timeline,
        "timeline_binned": base_bins,
        "peaks": peaks,
        "result": {
            "label": fusion["label"],
            "ai_score": fusion["ai_score"],
            "confidence": fusion["confidence"],
            "reason": fusion_core.get("reasons", []),
            "tips": tips,
        },
        "analysis_mode": "heuristic",
        "detector_version": DETECTOR_VERSION,
    }

@app.get("/healthz")
def healthz():
    ffprobe_ok = bool(shutil.which("ffprobe"))
    exiftool_ok = bool(shutil.which("exiftool"))
    ok = ffprobe_ok and exiftool_ok
    return JSONResponse({"ok": ok, "ffprobe": ffprobe_ok, "exiftool": exiftool_ok, "version": DETECTOR_VERSION},
                        status_code=200 if ok else 500)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file: _raise_422("Fornire un file.")
    body = await file.read()
    if len(body) > MAX_UPLOAD_BYTES:
        _raise_422("File troppo grande (>50 MB). Taglia o usa 'Registra 10s'.")
    key = "bytes:" + _file_hash(body)
    cached = _disk_cached(key)
    if cached: return JSONResponse(cached)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "upload")[1] or ".mp4")
    try:
        tmp.write(body); tmp.flush(); tmp.close()
        out = _analyze_path(tmp.name)
        _disk_store(key, out, expire=3600)
        return JSONResponse(out)
    finally:
        try: os.remove(tmp.name)
        except: pass

@app.post("/analyze-url")
async def analyze_url(request: Request):
    url_in: Optional[str] = None
    try:
        data = await request.json()
        if isinstance(data, dict): url_in = (data.get("url") or "").strip()
    except: pass
    if not url_in:
        try:
            form = await request.form()
            if "url" in form: url_in = str(form.get("url") or "").strip()
        except: pass
    if not url_in:
        url_in = (request.query_params.get("url") or "").strip()
    if not url_in: _raise_422("Fornire 'url'.")
    if not _is_allowed_url(url_in):
        _raise_422("Dominio non supportato. Usa 'Carica file' o 'Registra 10s'.")

    key = "url:" + hashlib.sha256(url_in.encode("utf-8")).hexdigest()
    cached = _disk_cached(key)
    if cached: return JSONResponse(cached)

    if not USE_YTDLP:
        _raise_422("Richiede estrazione dal provider. Attiva USE_YTDLP=1 o usa 'Carica file'.")

    fpath, resolved = _ytdlp_resolve_and_download(url_in)
    try:
        out = _analyze_path(fpath, source_url=url_in, resolved_url=resolved)
        _disk_store(key, out, expire=900)
        return JSONResponse(out)
    finally:
        try: os.remove(fpath)
        except: pass

@app.post("/analyze-link")
async def analyze_link(request: Request):
    url_in: Optional[str] = None
    try:
        data = await request.json()
        if isinstance(data, dict): url_in = (data.get("url") or "").strip()
    except: pass
    if not url_in:
        try:
            form = await request.form()
            if "url" in form: url_in = str(form.get("url") or "").strip()
        except: pass
    if not url_in:
        url_in = (request.query_params.get("url") or "").strip()
    if not url_in: _raise_422("Fornire 'url'.")

    parsed = urlparse(url_in)
    meta = {"source_url": url_in, "hostname": parsed.hostname, "path": parsed.path, "query": parsed.query}
    out = {
        "ok": True,
        "meta": meta,
        "forensic": {"c2pa": {"present": False}},
        "result": {
            "label": "uncertain",
            "ai_score": 0.5,
            "confidence": 0.5,
            "reason": ["context_only"],
            "tips": ["Carica il file o usa un link pubblico per un’analisi completa."]
        },
        "analysis_mode": "heuristic",
        "detector_version": DETECTOR_VERSION
    }
    return JSONResponse(out)

@app.post("/predict")
def predict(file: UploadFile = File(None), url: Optional[str] = Form(None)):
    if file is not None:
        return analyze(file=file)
    if url:
        _raise_422("Usa /analyze-url per i link, oppure /analyze con 'file'.")
    _raise_422("Fornire 'file' oppure 'url'.")
