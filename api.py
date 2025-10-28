# app/api.py
import os
import io
import shutil
import tempfile
import traceback
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from app.analyzers import audio as audio_an
from app.analyzers import fusion as fusion_an

# --- Config da ENV ---
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))  # 50MB
RESOLVER_MAX_BYTES = int(os.getenv("RESOLVER_MAX_BYTES", str(120 * 1024 * 1024)))
ALLOWED_ORIGINS = [o.strip() for o in (os.getenv("ALLOWED_ORIGINS", "").split(",")) if o.strip()]

app = FastAPI(title="AI-Video Detector API")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Utils ---

def _tmpdir(prefix: str = "aiv_") -> str:
    return tempfile.mkdtemp(prefix=prefix)

def _cleanup_dir(path: str):
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

def _probe_meta(video_path: str) -> Dict[str, Any]:
    """
    Qui presupponiamo che tu abbia già una funzione altrove che estrae meta+forensic+video_timeline.
    Per mantenere compatibilità con la tua struttura, emuliamo il formato che stai usando in UI.
    Sostituisci questo stub con il tuo vero extractor, oppure lascia la tua implementazione com'era.
    """
    # *** IMPORTANTE ***
    # Se nel tuo progetto hai già un modulo `app.analyzers.video` con `analyze(path)`,
    # usa quello e restituisci il dict con le stesse chiavi mostrate nei log.
    from app.analyzers import video as video_an
    meta, video = video_an.analyze(video_path)
    # meta: { width, height, fps, duration, bit_rate, vcodec, acodec, format_name, source_url, resolved_url, forensic{...} }
    # video: { width, height, src_fps, duration, sampled_fps, frames_sampled, timeline[...], summary{...} }
    return meta, video

def _fuse(meta: Dict[str,Any], video: Dict[str,Any], audio: Dict[str,Any]) -> Dict[str, Any]:
    return fusion_an.fuse(video=video, audio=audio, meta=meta or {})

def _ok(payload: Dict[str, Any]) -> JSONResponse:
    return JSONResponse(payload, status_code=200)

def _fail(status: int, message: str, extra: Dict[str,Any] = None) -> JSONResponse:
    body = {"detail": {"error": message}}
    if extra:
        body["detail"].update(extra)
    return JSONResponse(body, status_code=status)

# --- Endpoints ---

@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Valida dimensione
    try:
        content = await file.read()
    except Exception as e:
        return _fail(400, f"Errore lettura file: {e}")
    if not content:
        return _fail(415, "File vuoto o non ricevuto")

    if len(content) > MAX_UPLOAD_BYTES:
        return _fail(413, f"File troppo grande (> {MAX_UPLOAD_BYTES} bytes)")

    work = _tmpdir()
    try:
        in_path = os.path.join(work, file.filename or "upload.bin")
        with open(in_path, "wb") as f:
            f.write(content)

        # --- ANALISI ---
        meta, video = _probe_meta(in_path)
        audio = audio_an.analyze(in_path)  # nuovo audio analyzer robusto

        # === FUSIONE ===
        fused = _fuse(meta=meta, video=video, audio=audio)

        # Risposta compatibile con UI v1.1.3 (stesse chiavi viste nei tuoi log)
        return _ok({
            "ok": True,
            "meta": meta,
            "video": video,
            "audio": audio,
            "hints": {
                "heavy_compression": {
                    "score": 1.0 if "heavy_compression" in fused.get("reason","") else 0.0,
                    "reason": "Forte compressione (blockiness/banding)." if "compressione" in fused.get("reason","").lower() else ""
                }
            },
            "result": {
                "label": fused["label"],
                "ai_score": fused["ai_score"],
                "confidence": fused["confidence"],
                "reason": fused["reason"]
            },
            "timeline_binned": fused["timeline_binned"],
            "peaks": fused.get("peaks", [])
        })

    except Exception as e:
        return _fail(500, "Errore interno", {"trace": traceback.format_exc(), "msg": str(e)})
    finally:
        _cleanup_dir(work)

@app.post("/analyze-url")
async def analyze_url(
    url: Optional[str] = Form(None),
    link: Optional[str] = Form(None),
    q:    Optional[str] = Form(None),
):
    """
    Mantiene compatibilità: accetta url|link|q via FormData.
    Qui presupponiamo che tu abbia già un resolver yt-dlp/httpx robusto come nelle tue versioni recenti.
    """
    the_url = (url or link or q or "").strip()
    if not the_url:
        return _fail(422, "Nessun URL fornito")

    work = _tmpdir()
    try:
        # Scarica/risolvi in in_path (usa il tuo modulo esistente)
        from app.utils import fetch  # <-- se il tuo progetto usa un altro modulo, sostituisci qui
        in_path, meta_src = fetch.download(the_url, work_dir=work, max_bytes=RESOLVER_MAX_BYTES)

        meta, video = _probe_meta(in_path)
        # incorpora origine se disponibile
        if meta_src:
            meta.update({k:v for k,v in meta_src.items() if k not in meta})

        audio = audio_an.analyze(in_path)
        fused = _fuse(meta=meta, video=video, audio=audio)

        return _ok({
            "ok": True,
            "meta": meta,
            "video": video,
            "audio": audio,
            "hints": {
                "heavy_compression": {
                    "score": 1.0 if "heavy_compression" in fused.get("reason","") else 0.0,
                    "reason": "Forte compressione (blockiness/banding)." if "compressione" in fused.get("reason","").lower() else ""
                }
            },
            "result": {
                "label": fused["label"],
                "ai_score": fused["ai_score"],
                "confidence": fused["confidence"],
                "reason": fused["reason"]
            },
            "timeline_binned": fused["timeline_binned"],
            "peaks": fused.get("peaks", [])
        })

    except fetch.NeedsCookies as e:
        return _fail(422, "DownloadError yt-dlp: login/cookie richiesti", {"needs_cookies": True, "hint": str(e)})
    except fetch.RateLimited as e:
        return _fail(429, "Rate limited dalla piattaforma", {"rate_limited": True, "hint": str(e)})
    except Exception as e:
        return _fail(500, "Errore interno", {"trace": traceback.format_exc(), "msg": str(e)})
    finally:
        _cleanup_dir(work)

@app.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    url:  Optional[str] = Form(None),
    link: Optional[str] = Form(None),
    q:    Optional[str] = Form(None),
):
    """
    End-point retro-compatibile:
    - Se arriva un file → come /analyze
    - Se arriva un URL → come /analyze-url
    """
    if file is not None:
        return await analyze(file=file)
    the_url = (url or link or q or "").strip()
    if the_url:
        return await analyze_url(url=the_url)
    return _fail(415, "File vuoto o non ricevuto")
