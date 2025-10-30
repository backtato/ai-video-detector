import os, json, tempfile, subprocess, asyncio, signal, math
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from app.analyzers import audio as audio_an
from app.analyzers import video as video_an
from app.analyzers import forensic as forensic_an
from app.analyzers import meta as meta_an
from app.analyzers import fusion as fusion_an
from app.analyzers import heuristics_v2 as heur_an

VERSION = os.getenv("SERVICE_VERSION", "1.2.2")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
REQUEST_TIMEOUT_S = int(os.getenv("REQUEST_TIMEOUT_S", "240"))
USE_YTDLP = os.getenv("USE_YTDLP", "0") == "1"

app = FastAPI()
allow_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins if allow_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def run_cmd(cmd, timeout_s=30, cwd=None, env=None):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True, cwd=cwd, env=env)
    try:
        out, err = p.communicate(timeout=timeout_s)
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception: pass
        return 124, "", f"Timed out after {timeout_s}s"

def ffprobe_json(path: str) -> Dict[str, Any]:
    rc, out, err = run_cmd(["ffprobe", "-hide_banner", "-v", "error",
                            "-print_format", "json", "-show_streams", "-show_format", path], timeout_s=15)
    if rc != 0: return {}
    try: return json.loads(out)
    except Exception: return {}

def basic_meta(j: Dict[str, Any]) -> Dict[str, Any]:
    w = h = 0; fps = dur = 0.0; br = 0; vcodec = acodec = fmt = None
    try:
        for st in j.get("streams", []):
            if st.get("codec_type") == "video":
                w = int(st.get("width") or 0); h = int(st.get("height") or 0)
                r = st.get("r_frame_rate") or st.get("avg_frame_rate") or "0/1"
                try: a, b = r.split("/"); fps = float(a) / max(1.0, float(b))
                except Exception: fps = float(st.get("avg_frame_rate") or 0) or 0.0
                vcodec = st.get("codec_name")
                dur = float(st.get("duration") or j.get("format", {}).get("duration") or 0.0)
            elif st.get("codec_type") == "audio":
                acodec = st.get("codec_name")
        fmt = (j.get("format") or {}).get("format_name")
        br = int((j.get("format") or {}).get("bit_rate") or 0)
    except Exception: pass
    return dict(width=w, height=h, fps=fps, duration=dur, bit_rate=br, vcodec=vcodec, acodec=acodec, format_name=fmt)

def extract_wav_16k(path: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_path = tmp.name
    rc, out, err = run_cmd(["ffmpeg", "-hide_banner", "-y", "-i", path, "-ac", "1", "-ar", "16000", wav_path], timeout_s=45)
    if rc != 0:
        try: os.remove(wav_path)
        except Exception: pass
        return None, 0
    try:
        import soundfile as sf
        data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        os.remove(wav_path)
        return data, int(sr)
    except Exception:
        try: os.remove(wav_path)
        except Exception: pass
        return None, 0

def compression_hint(bit_rate: int, w: int, h: int, fps: float) -> str:
    if not bit_rate or not w or not h or not fps: return "unknown"
    bpp = bit_rate / max(1.0, (w*h*fps))
    if bpp < 0.05: return "heavy"
    if bpp < 0.12: return "normal"
    return "low"

def _analyze_file(path: str) -> Dict[str, Any]:
    meta_json = ffprobe_json(path)
    basic = basic_meta(meta_json)

    wav, sr = extract_wav_16k(path)
    duration_audio = float(len(wav) / sr) if (wav is not None and sr > 0) else 0.0
    audio = audio_an.analyze(wav, sr, duration_audio) if wav is not None else {
        "scores": {"audio_mean": 0.5}, "flags_audio": ["low_voice_presence"], "timeline": []
    }

    if not basic["duration"] or basic["duration"] == 0:
        if duration_audio > 0: basic["duration"] = duration_audio

    try:
        if (not basic["bit_rate"] or basic["bit_rate"] == 0) and basic["duration"] and basic["duration"] > 0:
            fsize_bits = os.path.getsize(path) * 8.0
            basic["bit_rate"] = int(fsize_bits / basic["duration"])
    except Exception: pass

    video = video_an.analyze(path, sample_seconds=int(min(30, max(5, int(basic.get("duration") or 15)))))
    v_summary = video.get("summary", {}) if isinstance(video, dict) else {}
    hints = {
        "bpp": None,
        "compression": compression_hint(basic["bit_rate"], basic["width"], basic["height"], basic["fps"]),
        "video_has_signal": bool(video.get("timeline")),
        "flow_used": float(v_summary.get("optflow_mag_avg") or 0.0),
        "motion_used": float(v_summary.get("motion_avg") or 0.0),
        "w": basic["width"], "h": basic["height"], "fps": basic["fps"], "br": basic["bit_rate"],
    }

    dev = meta_an.detect_device(path) if hasattr(meta_an, "detect_device") else {"vendor": None, "model": None, "os": None}
    if isinstance(dev, dict) and "device" in dev and isinstance(dev["device"], dict): dev = dev["device"]

    meta_out = {
        "width": basic["width"], "height": basic["height"], "fps": basic["fps"], "duration": basic["duration"],
        "bit_rate": basic["bit_rate"], "vcodec": basic["vcodec"], "acodec": basic["acodec"], "format_name": basic["format_name"],
        "source_url": None, "resolved_url": None,
        "forensic": forensic_an.analyze(path),
        "device": dev
    }

    fused = fusion_an.fuse(audio, video, hints, duration_sec=float(basic["duration"] or 0.0))

    out = {
        "ok": True,
        "meta": meta_out,
        "hints": hints,
        "result": fused["result"],
        "timeline_binned": fused["timeline_binned"],
        "peaks": fused["peaks"]
    }
    return out

@app.get("/healthz")
async def healthz():
    return JSONResponse({"ok": True, "ffprobe": True, "exiftool": True, "version": VERSION, "author": "Backtato"})

@app.get("/readyz")
async def readyz(): return JSONResponse({"ok": True, "version": VERSION})

@app.options("/{path:path}")
async def options_all(path: str): return PlainTextResponse("", status_code=204)

@app.api_route("/cors-test", methods=["GET", "POST", "OPTIONS"])
async def cors_test(): return JSONResponse({"ok": True})

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(status_code=415, detail={"error": "File vuoto o non ricevuto"})
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
        try:
            total = 0
            while True:
                chunk = await file.read(1024*1024)
                if not chunk: break
                tmp.write(chunk); total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    os.unlink(tmp.name)
                    return JSONResponse(status_code=413, content={"detail": f"File troppo grande (> {MAX_UPLOAD_BYTES} bytes)"})
            path = tmp.name
        except Exception as e:
            return JSONResponse(status_code=400, content={"detail": "Errore lettura file", "error": str(e)})
    try:
        result = await asyncio.wait_for(asyncio.to_thread(_analyze_file, path), timeout=min(REQUEST_TIMEOUT_S, 150))
        return JSONResponse(status_code=200, content=result)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"detail": "Analisi oltre il limite di tempo"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": "Analyze error", "error": str(e)})
    finally:
        try: os.remove(path)
        except Exception: pass

@app.post("/analyze-url")
async def analyze_url(url: str = Form(None), link: str = Form(None), q: str = Form(None)):
    target = url or link or q
    if not target: raise HTTPException(status_code=422, detail="Param url/link/q mancante")
    if not USE_YTDLP:
        return JSONResponse(status_code=415, content={"detail": "Per URL social abilita USE_YTDLP=1 oppure usa Carica file / Registra 10s"})
    with tempfile.TemporaryDirectory() as td:
        rc, out, err = run_cmd(["yt-dlp", "-f", "bv*+ba/best", "--no-playlist",
                                "--socket-timeout", "15", "--retries", "2", "--fragment-retries", "1",
                                "-o", f"{td}/vid.%(ext)s", target], timeout_s=60)
        if rc != 0: return JSONResponse(status_code=422, content={"detail": "DownloadError", "yt_dlp": err[:400]})
        vids = [p for p in os.listdir(td) if p.startswith("vid.")]
        if not vids: return JSONResponse(status_code=415, content={"detail": "Nessun media scaricato"})
        path = os.path.join(td, vids[0])
        try:
            result = await asyncio.wait_for(asyncio.to_thread(_analyze_file, path), timeout=min(REQUEST_TIMEOUT_S, 150))
            return JSONResponse(status_code=200, content=result)
        except asyncio.TimeoutError:
            return JSONResponse(status_code=504, content={"detail": "Analisi oltre il limite di tempo"})
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": "Analyze error", "error": str(e)})

@app.post("/predict")
async def predict(file: UploadFile = File(None), url: str = Form(None)):
    if file is not None: return await analyze(file)
    if url: return await analyze_url(url=url)
    raise HTTPException(status_code=422, detail="Fornisci file o url")

from fastapi.exceptions import RequestValidationError
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": "Validation error", "errors": exc.errors()})

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error", "error": str(exc)})
