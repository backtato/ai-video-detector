import os
import io
import re
import json
import shutil
import tempfile
import traceback
import subprocess
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

# === Tentativo opzionale: OpenCV per fallback FPS ===
try:
    import cv2  # opencv-python-headless
except Exception:
    cv2 = None

# === Config da ENV ===
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))  # 100MB
RESOLVER_MAX_BYTES = int(os.getenv("RESOLVER_MAX_BYTES", str(120 * 1024 * 1024)))
REQUEST_TIMEOUT_S = int(os.getenv("REQUEST_TIMEOUT_S", "120"))
USE_YTDLP = os.getenv("USE_YTDLP", "1") not in ("0", "false", "False", "")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGIN_REGEX = os.getenv("ALLOWED_ORIGIN_REGEX", "")

# === App ===
app = FastAPI(title="AI-Video Backend", version="1.1.3", docs_url=None, redoc_url=None)

# CORS
if ALLOWED_ORIGINS or ALLOWED_ORIGIN_REGEX:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()] if ALLOWED_ORIGINS else [],
        allow_origin_regex=ALLOWED_ORIGIN_REGEX or None,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# === Import moduli di analisi ===
try:
    from app.analyzers import meta as meta_mod
except Exception:
    meta_mod = None

try:
    from app.analyzers import audio as audio_mod
except Exception:
    audio_mod = None

try:
    from app.analyzers import video as video_mod
except Exception:
    video_mod = None

try:
    from app.analyzers import heuristics_v2 as heur_v2
except Exception:
    heur_v2 = None

try:
    from app import fusion as fusion_mod
except Exception:
    fusion_mod = None

# === Utils ===

USER_AGENT = "Mozilla/5.0 (AI-Video/1.1.3; +https://greensovereignfund.ch)"

def _bad_request(msg: str, extra: Dict[str, Any] = None):
    payload = {"detail": {"error": msg}}
    if extra:
        payload["detail"].update(extra)
    raise HTTPException(status_code=415, detail=payload["detail"])

def _unprocessable(msg: str, extra: Dict[str, Any] = None):
    payload = {"detail": {"error": msg}}
    if extra:
        payload["detail"].update(extra)
    raise HTTPException(status_code=422, detail=payload["detail"])

def _save_upload_to_temp(up: UploadFile) -> str:
    suffix = os.path.splitext(up.filename or "")[-1] or ".bin"
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    total = 0
    try:
        while True:
            chunk = up.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                up.file.close()
                tmpf.close()
                os.unlink(tmpf.name)
                _bad_request("File troppo grande", {"max_upload_bytes": MAX_UPLOAD_BYTES})
            tmpf.write(chunk)
    finally:
        up.file.close()
        tmpf.flush()
        tmpf.close()
    if total == 0:
        _bad_request("File vuoto o non ricevuto")
    return tmpf.name

def _run_ffprobe_json(path: str) -> Dict[str, Any]:
    # Fallback leggero se meta_mod non esiste
    try:
        cmd = [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=30)
        return json.loads(out.decode("utf-8", errors="ignore"))
    except Exception as e:
        return {"error": str(e)}

def _has_av_streams(ffmeta: Dict[str, Any]) -> bool:
    try:
        for s in ffmeta.get("streams", []):
            if s.get("codec_type") in ("video", "audio"):
                return True
    except Exception:
        pass
    return False

def _download_via_httpx(url: str, dst_path: str, limit_bytes: int = RESOLVER_MAX_BYTES) -> Dict[str, Any]:
    import httpx
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    with httpx.stream("GET", url, headers=headers, timeout=REQUEST_TIMEOUT_S, follow_redirects=True) as r:
        ct = r.headers.get("content-type", "")
        if "text/html" in ct.lower():
            return {"ok": False, "error": "HTML page (login/protected or not a media)", "content_type": ct}
        if "application/vnd.apple.mpegurl" in ct.lower() or "application/x-mpegURL" in ct.lower() or "mpegurl" in ct.lower():
            return {"ok": False, "error": "HLS stream not supported", "content_type": ct}

        total = 0
        with open(dst_path, "wb") as f:
            for chunk in r.iter_bytes():
                if not chunk:
                    break
                total += len(chunk)
                if total > limit_bytes:
                    return {"ok": False, "error": "Resolver max bytes exceeded", "bytes": total, "limit": limit_bytes}
                f.write(chunk)
    return {"ok": True, "bytes": total, "content_type": ct}

def _download_via_ytdlp(url: str, dst_dir: str) -> Dict[str, Any]:
    """
    Scarica media con yt-dlp (senza cookie). Ritorna:
      { ok, path?, error?, platform, warnings, needs_cookies, rate_limited, login_required }
    """
    import yt_dlp
    warnings = []
    platform = "generic"

    class _Logger:
        def debug(self, msg):  # yt-dlp log verboso
            pass
        def info(self, msg):
            if isinstance(msg, str) and "WARNING:" in msg:
                warnings.append(msg)
        def warning(self, msg):
            warnings.append(str(msg))
        def error(self, msg):
            warnings.append(f"ERROR: {msg}")

    ydl_opts = {
        "outtmpl": os.path.join(dst_dir, "dl.%(ext)s"),
        "format": "mp4/bestaudio+bestvideo/best",
        "noplaylist": True,
        "quiet": True,
        "nocheckcertificate": True,
        "retries": 2,
        "http_headers": {"User-Agent": USER_AGENT},
        "logger": _Logger(),
    }

    # piattaforma (solo per hint)
    try:
        if "instagram.com" in url:
            platform = "instagram"
        elif "youtube.com" in url or "youtu.be" in url:
            platform = "youtube"
        elif "tiktok.com" in url:
            platform = "tiktok"
        elif "facebook.com" in url:
            platform = "facebook"
    except Exception:
        pass

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            fn = ydl.prepare_filename(info)
            if os.path.exists(fn):
                return {"ok": True, "path": fn, "platform": platform, "warnings": warnings,
                        "needs_cookies": False, "rate_limited": False, "login_required": False}
            # fallback: cerca qualsiasi file creato
            for name in os.listdir(dst_dir):
                p = os.path.join(dst_dir, name)
                if name.startswith("dl.") and os.path.isfile(p):
                    return {"ok": True, "path": p, "platform": platform, "warnings": warnings,
                            "needs_cookies": False, "rate_limited": False, "login_required": False}
            return {"ok": False, "error": "File non trovato dopo download",
                    "platform": platform, "warnings": warnings,
                    "needs_cookies": False, "rate_limited": False, "login_required": False}
    except Exception as e:
        msg = str(e)
        needs_cookies = any(x in msg.lower() for x in [
            "use --cookies", "cookies", "sign in", "login required",
            "confirm you’re not a bot", "confirm you're not a bot",
            "adult content", "age-restricted"
        ])
        rate_limited = any(x in msg.lower() for x in [
            "rate limit", "too many requests", "quota exceeded", "429", "temporary unavailable"
        ])
        login_required = any(x in msg.lower() for x in [
            "login required", "private", "not available", "owner has restricted"
        ])
        return {"ok": False, "error": f"DownloadError yt-dlp: {msg}",
                "platform": platform, "warnings": warnings,
                "needs_cookies": needs_cookies,
                "rate_limited": rate_limited,
                "login_required": login_required}

# === Fallback meta summary (FPS/size/duration/bitrate) ===

def _safe_num(x):
    try:
        return float(x)
    except Exception:
        return None

def _estimate_fps_via_opencv(path: str) -> Optional[float]:
    """
    Ultimo fallback: prova a stimare l'FPS con OpenCV senza leggere tutto il file.
    1) CAP_PROP_FPS se sensato (1 < fps < 120)
    2) altrimenti legge timestamp dei primi ~2 secondi e usa la mediana dei delta
    """
    if cv2 is None:
        return None
    try:
        cap = cv2.VideoCapture(path)
        try:
            fps_prop = cap.get(cv2.CAP_PROP_FPS)
            if fps_prop and fps_prop > 1.0 and fps_prop < 120.0:
                return float(fps_prop)
            # stima via timestamp
            timestamps = []
            ms0 = None
            frames = 0
            # leggi al massimo 120 frame o 2 secondi
            while frames < 120:
                ok, _ = cap.read()
                if not ok:
                    break
                ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if ms is None:
                    break
                if ms0 is None:
                    ms0 = ms
                dt = ms - ms0
                if dt > 2000:  # ~2s
                    break
                timestamps.append(ms)
                frames += 1
            if len(timestamps) > 5:
                deltas = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:]) if (t2 - t1) > 0.5]
                if deltas:
                    # mediana dei delta → fps
                    deltas_sorted = sorted(deltas)
                    mid = len(deltas_sorted) // 2
                    if len(deltas_sorted) % 2 == 1:
                        med = deltas_sorted[mid]
                    else:
                        med = 0.5 * (deltas_sorted[mid - 1] + deltas_sorted[mid])
                    if med > 0:
                        return 1000.0 / med
        finally:
            cap.release()
    except Exception:
        return None
    return None

def _extract_video_summary_from_ffprobe(ff: dict) -> dict:
    """Fallback: crea un summary con width/height/fps/duration/bit_rate da ffprobe (robusto per iPhone/QuickTime)."""
    out = {}
    fps_source = None
    try:
        fmt = ff.get("format", {}) if isinstance(ff, dict) else {}
        if "bit_rate" in fmt:
            try:
                out["bit_rate"] = int(fmt["bit_rate"])
            except Exception:
                pass
        if "duration" in fmt:
            d = _safe_num(fmt.get("duration"))
            if d:
                out["duration"] = float(d)

        vstreams = [s for s in ff.get("streams", []) if s.get("codec_type") == "video"]
        if vstreams:
            vs = vstreams[0]
            w = vs.get("width")
            h = vs.get("height")
            if w: out["width"] = int(w)
            if h: out["height"] = int(h)

            # 1) Prova nb_frames / duration (spesso assente su iPhone, ma se c'è è affidabile)
            nb = vs.get("nb_frames")
            if nb and "duration" in out:
                try:
                    nb = float(nb)
                    if nb > 1 and out["duration"] > 0:
                        out["fps"] = nb / out["duration"]
                        fps_source = "ffprobe_nbframes"
                except Exception:
                    pass

            # 2) Prova r_frame_rate o avg_frame_rate (evita 0/0)
            if "fps" not in out:
                r = vs.get("r_frame_rate")
                if isinstance(r, str) and "/" in r and r != "0/0":
                    try:
                        a, b = r.split("/")
                        a, b = float(a), float(b)
                        if b > 0 and 1.0 < (a/b) < 240.0:
                            out["fps"] = a / b
                            fps_source = "ffprobe_r"
                    except Exception:
                        pass
            if "fps" not in out:
                r = vs.get("avg_frame_rate")
                if isinstance(r, str) and "/" in r and r != "0/0":
                    try:
                        a, b = r.split("/")
                        a, b = float(a), float(b)
                        if b > 0 and 1.0 < (a/b) < 240.0:
                            out["fps"] = a / b
                            fps_source = "ffprobe_avg"
                    except Exception:
                        pass

            # 3) Ultimo tentativo: OpenCV
            if "fps" not in out:
                est = _estimate_fps_via_opencv(ff.get("_input_path") or fmt.get("filename") or "")
                if est and 1.0 < est < 240.0:
                    out["fps"] = float(est)
                    fps_source = "opencv_prop"  # o "opencv_ts" (non distinguiamo qui)
    except Exception:
        pass

    if fps_source:
        out["fps_source"] = fps_source
    return out

def _enrich_meta_summary(meta: dict):
    """Se manca meta['summary'].fps/width/height/duration/bit_rate, prova a ricavarli da ffprobe + OpenCV."""
    if not isinstance(meta, dict):
        return
    summary = meta.get("summary")
    ff = meta.get("ffprobe", {})
    if not summary:
        summary = {}
    need = any(k not in summary for k in ("width", "height", "fps", "duration", "bit_rate"))
    # per permettere a _extract_video_summary_from_ffprobe di usare OpenCV,
    # proviamo a inserire il filename nel dict ff (se non presente)
    if isinstance(ff, dict) and "_input_path" not in ff:
        try:
            # meta_mod di solito non mette il filename; lo passiamo noi via campo privato
            if meta.get("source_path"):
                ff["_input_path"] = meta["source_path"]
        except Exception:
            pass
    if need and ff:
        fallback = _extract_video_summary_from_ffprobe(ff)
        for k, v in fallback.items():
            if k not in summary and v is not None:
                summary[k] = v
    meta["summary"] = summary

def _analyze_file(path: str, source_url: Optional[str] = None) -> Dict[str, Any]:
    # 1) Meta
    if meta_mod and hasattr(meta_mod, "get_media_meta"):
        meta = meta_mod.get_media_meta(path, source_url=source_url)
        ff = meta.get("ffprobe", {})
        # memorizza per OpenCV fallback (vedi _enrich_meta_summary)
        meta["source_path"] = path
    else:
        ff = _run_ffprobe_json(path)
        ff["_input_path"] = path
        meta = {"ffprobe": ff, "source_path": path}

    # Enrich summary (FPS/size/duration/bitrate fallback)
    try:
        _enrich_meta_summary(meta)
    except Exception:
        pass

    if not _has_av_streams(ff):
        _bad_request("Unsupported or empty media: no audio/video streams found",
                     {"ffprobe_found": True, "file_size": os.path.getsize(path)})

    # 2) Video
    video_stats = None
    if video_mod and hasattr(video_mod, "analyze"):
        try:
            video_stats = video_mod.analyze(path, max_seconds=30.0, fps=2.5)
        except Exception as e:
            video_stats = {"error": f"video_analyze_error: {e}"}

    # 3) Audio
    audio_stats = None
    if audio_mod and hasattr(audio_mod, "analyze"):
        try:
            audio_stats = audio_mod.analyze(path, target_sr=16000)
        except Exception as e:
            audio_stats = {"error": f"audio_analyze_error: {e}"}

    # 4) Hints / Heuristics v2
    hints = {}
    if heur_v2 and hasattr(heur_v2, "compute_hints"):
        try:
            hints = heur_v2.compute_hints(meta=meta, video_stats=video_stats, audio_stats=audio_stats)
        except Exception as e:
            hints = {"error": f"hints_error: {e}"}

    # 5) Fusione
    if fusion_mod and hasattr(fusion_mod, "fuse"):
        try:
            fused = fusion_mod.fuse(video_stats=video_stats, audio_stats=audio_stats, hints=hints, meta=meta)
        except Exception as e:
            fused = {"result": {"label": "uncertain", "ai_score": 0.5, "confidence": 0.4, "reason": f"fusion_error: {e}"}}
    else:
        fused = {"result": {"label": "uncertain", "ai_score": 0.5, "confidence": 0.4, "reason": "fusion_missing"}}

    out = {
        "ok": True,
        "meta": meta.get("summary", meta.get("format", {})) or {},
        "forensic": meta.get("forensic", {}),
        "video": video_stats or {},
        "audio": audio_stats or {},
        "hints": hints or {},
        **fused
    }
    return out

# === Routes ===

@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not isinstance(file, (UploadFile, StarletteUploadFile)):
        _bad_request("Nessun file ricevuto")
    path = _save_upload_to_temp(file)
    try:
        res = _analyze_file(path, source_url=None)
        return JSONResponse(res)
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass

@app.post("/analyze-url")
async def analyze_url(request: Request, url: Optional[str] = Form(None)):
    # Accetta FormData, JSON o query (?url= / ?link= / ?q=)
    the_url = url
    if not the_url:
        try:
            if request.headers.get("content-type", "").startswith("application/json"):
                data = await request.json()
                the_url = data.get("url") or data.get("link") or data.get("q")
        except Exception:
            pass
    if not the_url:
        qs = dict(request.query_params)
        the_url = qs.get("url") or qs.get("link") or qs.get("q")

    if not the_url or not re.match(r"^https?://", the_url):
        _bad_request("URL non valido")

    tmpdir = tempfile.mkdtemp(prefix="aivdl_")
    path = os.path.join(tmpdir, "media.bin")
    used = "none"

    try:
        if USE_YTDLP:
            ytd = _download_via_ytdlp(the_url, tmpdir)
            if ytd.get("ok"):
                path = ytd["path"]
                used = "yt-dlp"
            else:
                # mappa errori noti in 422 con hint specifici
                plat = ytd.get("platform", "generic")
                if ytd.get("needs_cookies") or ytd.get("login_required"):
                    _unprocessable("Contenuto protetto o con restrizioni (autenticazione richiesta).", {
                        "platform": plat,
                        "hint": "Il contenuto richiede login/cookie. Usa 'Carica file' o 'Registra 10s e carica'."
                    })
                if ytd.get("rate_limited"):
                    _unprocessable("Sorgente ha applicato un rate limit (anti-bot).", {
                        "platform": plat,
                        "hint": "Riprova più tardi o usa 'Carica file' / 'Registra 10s e carica'."
                    })
                # fallback httpx per URL diretti
                dl = _download_via_httpx(the_url, path, RESOLVER_MAX_BYTES)
                if not dl.get("ok"):
                    _unprocessable(dl.get("error", "Impossibile scaricare il media"), {
                        "platform": plat,
                        "hint": "Se è un link social protetto/login-wall, usa 'Carica file' o 'Registra 10s e carica'."
                    })
                used = "httpx"
        else:
            dl = _download_via_httpx(the_url, path, RESOLVER_MAX_BYTES)
            if not dl.get("ok"):
                _unprocessable(dl.get("error", "Impossibile scaricare il media"),
                               {"hint": "Se è protetto/login-wall, usa Carica file o Registra 10s"})
            used = "httpx"

        if not os.path.exists(path) or os.path.getsize(path) == 0:
            _unprocessable("Download completato ma file mancante o vuoto",
                           {"resolver": used})

        res = _analyze_file(path, source_url=the_url)
        res.setdefault("tips", {})
        res["tips"]["resolver"] = used
        if 'ytd' in locals() and isinstance(ytd, dict):
            if ytd.get("warnings"):
                res["tips"]["ytdlp_warnings"] = ytd["warnings"][:5]
            if ytd.get("platform"):
                res["tips"]["platform"] = ytd["platform"]
        return JSONResponse(res)

    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

@app.post("/analyze-link")
async def analyze_link(link: str = Form(...)):
    # Endpoint "contestuale": non scarica; restituisce placeholder guida
    return JSONResponse({
        "ok": True,
        "meta": {},
        "forensic": {},
        "result": {
            "label": "uncertain",
            "ai_score": 0.5,
            "confidence": 0.4,
            "reason": "Solo analisi del contesto/link: fornisci il video o usa /analyze-url per il download."
        },
        "tips": {
            "guide": "Per link protetti (Instagram privati, storie, ecc.), usa 'Carica file' o 'Registra 10s e carica'."
        }
    })

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(None), url: Optional[str] = Form(None)):
    """
    Retro-compatibilità:
    - se c'è un file NON VUOTO -> /analyze
    - altrimenti prova a prendere l'URL da Form, JSON o query -> /analyze-url
    """
    # 1) Se è presente un file, verifichiamo che non sia vuoto
    if file and getattr(file, "filename", None):
        try:
            peek = await file.read(16)
            await file.seek(0)
        except Exception:
            peek = None
        if peek:
            return await analyze(file=file)
        # se è vuoto, ignoriamo il file e proviamo l'URL

    # 2) URL da Form
    the_url = url
    # 3) Fallback JSON
    if not the_url:
        try:
            if request.headers.get("content-type", "").startswith("application/json"):
                data = await request.json()
                the_url = data.get("url") or data.get("link") or data.get("q")
        except Exception:
            pass
    # 4) Fallback querystring
    if not the_url:
        qs = dict(request.query_params)
        the_url = qs.get("url") or qs.get("link") or qs.get("q")

    if the_url:
        return await analyze_url(request=request, url=the_url)

    _bad_request("Nessun file o URL fornito")
