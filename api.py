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

__author__ = "Backtato"  # NEW

APP_NAME = "AI-Video Detector"
DETECTOR_VERSION = os.getenv("DETECTOR_VERSION", "1.2.0")

# CHANGED: default 100 MB; parametrico
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
RESOLVER_MAX_BYTES = int(os.getenv("RESOLVER_MAX_BYTES", str(120 * 1024 * 1024)))
ALLOWED_ORIGINS = [o for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o]
USE_YTDLP = os.getenv("USE_YTDLP", "1") not in ("0", "false", "False")
RESOLVER_UA = os.getenv("RESOLVER_UA", "Mozilla/5.0 (compatible; AI-Video/1.0)")
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/aivideo-cache")

# --- i18n minimale (puoi estendere) ---  # NEW
_I18N = {
    "it": {
        "err_provide_file": "Fornire un file.",
        "err_too_big": "File troppo grande (>{limit_mb} MB). Taglia o usa 'Registra 10s'.",
        "err_provide_url": "Fornire 'url'.",
        "err_use_analyze_url": "Usa /analyze-url per i link, oppure /analyze con 'file'.",
        "err_protected": "Contenuto protetto o login richiesto. Suggerimento: usa 'Carica file' o 'Registra 10s'.",
        "tip_uncertain": "Esito incerto: prova un video più lungo o di qualità migliore.",
        "tip_no_c2pa": "Nessuna firma C2PA rilevata (non prova di falsità).",
        "tip_audio_pauses": "Audio con molte pause o a bassa energia."
    },
    "en": {
        "err_provide_file": "Please provide a file.",
        "err_too_big": "File too large (>{limit_mb} MB). Trim or use 'Record 10s'.",
        "err_provide_url": "Provide 'url'.",
        "err_use_analyze_url": "Use /analyze-url for links, or /analyze with 'file'.",
        "err_protected": "Protected content or login required. Tip: use 'Upload file' or 'Record 10s'.",
        "tip_uncertain": "Uncertain result: try a longer or higher-quality video.",
        "tip_no_c2pa": "No C2PA signature detected (not proof of falsity).",
        "tip_audio_pauses": "Audio with long pauses or low energy."
    },
    "fr": {
        "err_provide_file": "Veuillez fournir un fichier.",
        "err_too_big": "Fichier trop volumineux (>{limit_mb} Mo). Coupez-le ou utilisez 'Enregistrer 10s'.",
        "err_provide_url": "Fournissez 'url'.",
        "err_use_analyze_url": "Utilisez /analyze-url pour les liens, ou /analyze avec 'file'.",
        "err_protected": "Contenu protégé ou connexion requise. Astuce : 'Téléverser un fichier' ou 'Enregistrer 10s'.",
        "tip_uncertain": "Résultat incertain : essayez une vidéo plus longue ou de meilleure qualité.",
        "tip_no_c2pa": "Aucune signature C2PA détectée (pas une preuve de falsification).",
        "tip_audio_pauses": "Audio avec de longues pauses ou faible énergie."
    },
    "de": {
        "err_provide_file": "Bitte eine Datei bereitstellen.",
        "err_too_big": "Datei zu groß (>{limit_mb} MB). Kürzen oder '10s aufnehmen' verwenden.",
        "err_provide_url": "'url' angeben.",
        "err_use_analyze_url": "Für Links /analyze-url nutzen oder /analyze mit 'file'.",
        "err_protected": "Geschützter Inhalt oder Anmeldung erforderlich. Tipp: 'Datei hochladen' oder '10s aufnehmen'.",
        "tip_uncertain": "Unklarer Befund: Versuchen Sie ein längeres oder qualitativ besseres Video.",
        "tip_no_c2pa": "Keine C2PA-Signatur erkannt (kein Beweis für Fälschung).",
        "tip_audio_pauses": "Audio mit vielen Pausen oder geringer Energie."
    }
}
_DEFAULT_LANG = "it"

def _negotiate_lang(request: Optional[Request]) -> str:
    # 1) query ?lang=xx 2) X-Lang 3) Accept-Language 4) default
    try:
        if request is None:
            return _DEFAULT_LANG
        q = (request.query_params.get("lang") or "").strip().lower()
        if q in _I18N: return q
        xl = (request.headers.get("X-Lang") or "").split(",")[0].strip().lower()
        if xl in _I18N: return xl
        al = (request.headers.get("Accept-Language") or "").lower()
        # es. "it-CH,it;q=0.9,en;q=0.8"
        for token in [t.split(";")[0].strip() for t in al.split(",") if t]:
            base = token.split("-")[0]
            if base in _I18N: return base
    except:
        pass
    return _DEFAULT_LANG

def _t(lang: str, key: str, **kw) -> str:
    s = (_I18N.get(lang) or {}).get(key) or (_I18N[_DEFAULT_LANG].get(key) or key)
    if kw:
        try: return s.format(**kw)
        except: return s
    return s

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

app = FastAPI(title=APP_NAME, version=DETECTOR_VERSION)

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
        return u.scheme in ("http", "https") and any((u.netloc or "").lower().endswith(d) for d in ALLOW_DOMAINS)
    except:
        return False

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
            if not info:
                _raise_422("Download fallito.")
            # trova file scaricato
            fn = ydl.prepare_filename(info)
            if not os.path.exists(fn):
                # prova cercando nel tmpdir
                cand = None
                for root, _, files in os.walk(tmpdir):
                    for f in files:
                        if f.endswith((".mp4",".mov",".mkv",".webm",".m4v",".avi")):
                            cand = os.path.join(root, f); break
                    if cand: break
                fn = cand
            if not fn or not os.path.exists(fn):
                _raise_422("Download riuscito ma file non trovato.")
            return fn, (info.get("url") or url)
    except Exception as e:
        msg = str(e).lower()
        if any(x in msg for x in ["login", "private", "cookies", "not available", "rate", "protected"]):
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

    hints = build_hints(meta, v_res.get("flags_video"), c2pa_present=c2pa.get("present", False), dev_fp=dev_fp)
    fusion_core = fuse_scores(
        frame=v_res["scores"].get("frame_mean", 0.5),
        audio=a_res["scores"].get("audio_mean", 0.5),
        hints=hints
    )

    timeline = (v_res.get("timeline") or []) + (a_res.get("timeline") or [])
    duration = meta.get("duration") or 0.0
    base_bins = _bin_timeline(timeline, duration=duration or 0.0, bin_sec=1.0, mode="max")
    fusion, peaks = finalize_with_timeline(fusion_core, base_bins)

    tips = []
    if fusion["label"] == "uncertain": tips.append("tip_uncertain")
    if hints.get("no_c2pa"): tips.append("tip_no_c2pa")
    if "long_pauses" in (a_res.get("flags_audio") or []): tips.append("tip_audio_pauses")

    out = {
        "ok": True,
        "meta": meta,
        "forensic": {
            "c2pa": {"present": c2pa.get("present", False)},
            "device": dev_fp,
        },
        "result": {
            "label": fusion["label"],
            "ai_score": float(fusion["ai_score"]),
            "confidence": float(fusion["confidence"]),
            "reasons": fusion.get("reasons") or [],
            "timeline": base_bins,
            "peaks": peaks,
            "tips": tips,  # chiavi i18n che il frontend può tradurre, o backend può risolvere
        },
        "analysis_mode": "heuristic",
        "detector_version": DETECTOR_VERSION,
    }
    return out

@app.get("/healthz")
def healthz():
    ffprobe_ok = bool(shutil.which("ffprobe"))
    exiftool_ok = bool(shutil.which("exiftool"))
    ok = ffprobe_ok and exiftool_ok
    return JSONResponse({"ok": ok, "ffprobe": ffprobe_ok, "exiftool": exiftool_ok,
                         "version": DETECTOR_VERSION, "author": __author__}, status_code=200 if ok else 500)

# CHANGED: upload a chunk + i18n error messages
@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...)):
    lang = _negotiate_lang(request)
    if not file:
        _raise_422(_t(lang, "err_provide_file"))

    # cache by file hash is tricky without full read; usiamo temp+hash progressivo opzionale
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "upload")[1] or ".mp4")
    size = 0
    sha = hashlib.sha256()
    try:
        while True:
            chunk = await file.read(1024 * 1024)  # 1 MB
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                tmp.close()
                try: os.remove(tmp.name)
                except: pass
                _raise_422(_t(lang, "err_too_big", limit_mb=int(MAX_UPLOAD_BYTES/1024/1024)))
            sha.update(chunk)
            tmp.write(chunk)
        tmp.flush(); tmp.close()

        key = "bytes:" + sha.hexdigest()
        cached = _disk_cached(key)
        if cached:
            # risolvi eventuali tip i18n lato backend (opzionale)
            if isinstance(cached, dict) and "result" in cached and "tips" in cached["result"]:
                cached_local = json.loads(json.dumps(cached))  # copia profonda
                cached_local["i18n_lang"] = lang
                # traduzione tips (chiavi) -> stringhe
                if isinstance(cached_local["result"].get("tips"), list):
                    cached_local["result"]["tips"] = [
                        _t(lang, t) if t in (_I18N.get(lang) or {}) else t
                    for t in cached_local["result"]["tips"]
                ]
                return JSONResponse(cached_local)
            return JSONResponse(cached)

        out = _analyze_path(tmp.name)
        # applica i18n alle tips
        if isinstance(out.get("result", {}).get("tips"), list):
            out["i18n_lang"] = lang
            out["result"]["tips"] = [
                _t(lang, t) if t in (_I18N.get(lang) or {}) else t
            for t in out["result"]["tips"]
        ]
        _disk_store(key, out, expire=3600)
        return JSONResponse(out)
    finally:
        # il file temp verrà comunque cancellato: qui lo lasciamo perché serve per timeline? No: già analizzato.
        try: os.remove(tmp.name)
        except: pass

@app.post("/analyze-link")
async def analyze_link(request: Request):
    lang = _negotiate_lang(request)
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
    if not url_in:
        _raise_422(_t(lang, "err_provide_url"))

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
            "reasons": ["context_only"],
            "tips": [_t(lang, "err_use_analyze_url")],
        },
        "analysis_mode": "heuristic",
        "detector_version": DETECTOR_VERSION,
        "i18n_lang": lang
    }
    return JSONResponse(out)

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(None), url: Optional[str] = Form(None)):
    lang = _negotiate_lang(request)
    if file is not None:
        return await analyze(request, file=file)
    if url:
        _raise_422(_t(lang, "err_use_analyze_url"))
    _raise_422(_t(lang, "err_provide_file"))
