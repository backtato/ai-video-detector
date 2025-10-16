from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os

from app.config import settings
from app.utils.video import read_video_frames
from app.detectors.baseline import BaselineDetector
from app.utils.download import download_video


app = FastAPI(title="AI Video Detector", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

detector = BaselineDetector(
    target_fps=settings.TARGET_FPS,
    max_frames=settings.MAX_FRAMES,
    min_edge_var=settings.MIN_EDGE_VAR,
)

class PredictOut(BaseModel):
    score: float
    label: str
    details: dict

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>AI Video Detector</title>
      <style>
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 40px; }
        form { display: grid; gap: 12px; max-width: 520px; }
        input, button { padding: 10px; font-size: 16px; }
        .hint { color: #555; }
      </style>
    </head>
    <body>
      <h1>AI Video Detector</h1>
      <p class="hint">Inserisci un <b>link</b> (YouTube/social) <i>oppure</i> carica un <b>file</b> (max 50MB).</p>
      <form action="/predict" method="post" enctype="multipart/form-data">
        <label>URL (YouTube/TikTok/X/Instagram/Facebook/Reddit)
          <input type="url" name="url" placeholder="https://www.youtube.com/watch?v=...">
        </label>
        <div>— oppure —</div>
        <label>File video
          <input type="file" name="file" accept=".mp4,.mov,.mkv,.webm,.avi">
        </label>
        <button type="submit">Analizza</button>
      </form>
      <p class="hint">Endpoint GET rapidi: <code>/predict-get?url=...</code> oppure <code>/predict?url=...</code></p>
    </body>
    </html>
    """

# POST classico (multipart) — usa form-data: url *oppure* file
@app.post("/predict", response_model=PredictOut)
async def predict(url: str | None = Form(default=None), file: UploadFile | None = File(default=None)):
    return await _predict_impl(url=url, file=file)

# GET comodo: /predict-get?url=...
@app.get("/predict-get", response_model=PredictOut)
async def predict_get(url: str | None = None):
    if not url:
        raise HTTPException(400, "Usa ?url=... per fornire un link YouTube/social o diretto al file.")
    return await _predict_impl(url=url, file=None)

# Alias GET: /predict?url=...
@app.get("/predict", response_model=PredictOut)
async def predict_alias(url: str | None = None):
    if not url:
        raise HTTPException(400, "Usa ?url=... per fornire un link YouTube/social o diretto al file.")
    return await _predict_impl(url=url, file=None)

async def _predict_impl(url: str | None, file: UploadFile | None):
    if not url and not file:
        raise HTTPException(400, "Fornisci un 'url' (YouTube/social o diretto) oppure carica un 'file'.")
    if url and file:
        raise HTTPException(400, "Invia o l'URL o il file, non entrambi.")

    tmp_path = None
    try:
        if url:
            tmp_path, source = download_video(url, settings.TMP_DIR, settings.MAX_UPLOAD_MB)
        else:
            if not file.filename.lower().endswith(settings.ALLOWED_EXTS):
                raise HTTPException(400, "Estensione non valida. Usa mp4/mov/mkv/webm/avi.")
            contents = await file.read()
            if len(contents) > settings.MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(413, f"File troppo grande: max {settings.MAX_UPLOAD_MB}MB.")
            os.makedirs(settings.TMP_DIR, exist_ok=True)
            tmp_path = os.path.join(settings.TMP_DIR, file.filename)
            with open(tmp_path, "wb") as f:
                f.write(contents)

        frames, fps, w, h = read_video_frames(
            tmp_path,
            target_fps=settings.TARGET_FPS,
            max_frames=settings.MAX_FRAMES
        )
        if len(frames) < 8:
            raise HTTPException(422, "Video troppo corto per un’analisi affidabile (min 8 frame).")

        score, details = detector.score(frames, fps=fps)
        label = "ai-generated" if score >= settings.THRESHOLD else "real"

        return PredictOut(score=round(float(score), 4), label=label, details=details)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Errore durante l'analisi: {e}")
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))