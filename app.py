import os, re, json, shlex, asyncio, subprocess, tempfile
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

APP_NAME = "ai-video-detector"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "50000000"))  # 50MB
RESOLVER_ALLOWLIST = os.getenv("RESOLVER_ALLOWLIST", "youtube.com,vimeo.com,cdn,mp4,m3u8")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "")  # opzionale: es. https://tuo-dominio

app = FastAPI(title=APP_NAME)

allow_origins = [o for o in [FRONTEND_ORIGIN] if o] or ["*"]  # sblocca subito; restringi appena hai il dominio
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------- Schemi ----------
class URLIn(BaseModel):
    url: str

def _verdict_from_probe(probe_json: dict) -> dict:
    """
    Heuristica placeholder: restituisce un JSON coerente con {label, ai_plausibility, confidence}
    Puoi sostituire qui con la tua logica/ML.
    """
    # euristica delicata e “safe”: se ffprobe vede almeno 1 stream video valido => "Inconclusive"
    streams = probe_json.get("streams") or []
    has_video = any(s.get("codec_type") == "video" for s in streams)
    if not has_video:
        return {"label": "Inconclusive", "ai_plausibility": 0.5, "confidence": 0.4}
    # banalissima metrica: bitrate o fps strani => più sospetto
    fps = 0.0
    for s in streams:
        if s.get("codec_type") == "video" and s.get("avg_frame_rate") and s["avg_frame_rate"] != "0/0":
            try:
                num, den = s["avg_frame_rate"].split("/")
                fps = float(num) / float(den)
            except Exception:
                pass
    if fps and (fps > 70 or fps < 12):
        return {"label": "AI-likely", "ai_plausibility": 0.72, "confidence": 0.6}
    return {"label": "Inconclusive", "ai_plausibility": 0.48, "confidence": 0.45}

async def _run_ffprobe(input_path_or_url: str, timeout: int = 25) -> dict:
    """
    Esegue ffprobe in JSON; supporta URL http(s) e file locali.
    """
    cmd = f'ffprobe -v error -print_format json -show_streams -show_format {shlex.quote(input_path_or_url)}'
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise HTTPException(status_code=408, detail="ffprobe timeout")
    if proc.returncode != 0:
        # Errore “classico” se URL non diretto o container senza permessi protocolli
        raise HTTPException(status_code=422, detail=f"ffprobe failed: {err.decode('utf-8', 'ignore')[:300]}")
    try:
        return json.loads(out.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="Invalid ffprobe JSON")

def _looks_direct_media(url: str) -> bool:
    return bool(re.search(r'\.(mp4|mov|webm|mkv|m3u8)(\?|$)', url, re.I))

# ---------- Endpoints ----------
@app.get("/")
def ping():
    return {"name": APP_NAME, "status": "ok"}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # size guard (se conosci content_length via client, altrimenti scriviamo su tmp e verifichiamo)
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (> {MAX_UPLOAD_BYTES} bytes)")
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename)[-1] or ".bin", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        probe = await _run_ffprobe(tmp.name)
    verdict = _verdict_from_probe(probe)
    return {
        "ok": True,
        "input": {"filename": file.filename, "bytes": len(data)},
        "probe": {"format": probe.get("format", {}), "streams": probe.get("streams", [])[:2]},  # riduci risposta
        **verdict,
    }

@app.post("/analyze-url")
async def analyze_url(body: URLIn):
    url = body.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="url required")
    if not _looks_direct_media(url):
        # non provare ffprobe: chiarisci subito l’errore
        raise HTTPException(status_code=400, detail="Not a direct media URL (.mp4/.m3u8 etc.). Use /analyze-link.")
    # opzionale: HEAD per dimensione
    try:
        head = requests.head(url, timeout=8, allow_redirects=True)
        cl = int(head.headers.get("content-length", "0") or "0")
        if cl and cl > MAX_UPLOAD_BYTES:
            # consentiamo comunque: è URL, non upload — ma informiamo
            pass
    except Exception:
        pass
    probe = await _run_ffprobe(url)
    verdict = _verdict_from_probe(probe)
    return {
        "ok": True,
        "input": {"url": url},
        "probe": {"format": probe.get("format", {}), "streams": probe.get("streams", [])[:2]},
        **verdict,
    }

@app.post("/analyze-link")
def analyze_link(body: URLIn):
    url = body.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="url required")
    # Per link tipo YouTube, restituiamo analisi “context-only” se non possiamo scaricare
    info = {"provider": None, "id": None, "title": None}
    if "youtube.com" in url or "youtu.be" in url:
        info["provider"] = "youtube"
        m = re.search(r'(?:v=|/)([A-Za-z0-9_-]{6,})', url)
        if m:
            info["id"] = m.group(1)
        if not YOUTUBE_API_KEY:
            return {
                "ok": True,
                "context_only": True,
                "message": "Provide YOUTUBE_API_KEY to enrich metadata; media not downloaded.",
                "input": {"url": url},
                "info": info,
                "label": "Inconclusive",
                "ai_plausibility": 0.5,
                "confidence": 0.3
            }
        # opzionale: chiamata YouTube Data API v3 (solo metadati)
        try:
            r = requests.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={"id": info["id"], "key": YOUTUBE_API_KEY, "part": "snippet,contentDetails"},
                timeout=8
            )
            if r.ok:
                data = r.json()
                if data.get("items"):
                    info["title"] = data["items"][0]["snippet"]["title"]
        except Exception:
            pass
        return {
            "ok": True,
            "context_only": True,
            "message": "Metadata fetched; media not downloaded.",
            "input": {"url": url},
            "info": info,
            "label": "Inconclusive",
            "ai_plausibility": 0.5,
            "confidence": 0.35
        }
    # Altri provider → solo contesto
    return {
        "ok": True, "context_only": True, "message": "Generic link; media not downloaded.",
        "input": {"url": url},
        "label": "Inconclusive", "ai_plausibility": 0.5, "confidence": 0.3
    }
