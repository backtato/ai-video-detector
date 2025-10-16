
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from app.config import settings
from app.utils.video import read_video_frames
from app.detectors.baseline import BaselineDetector
from app.utils.download import download_video

app = FastAPI(title="AI Video Detector", version="1.1.0")

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

@app.post("/predict", response_model=PredictOut)
async def predict(url: str | None = Form(default=None), file: UploadFile | None = File(default=None)):
    # Deve arrivare o url o file
    if not url and not file:
        raise HTTPException(400, "Fornisci un 'url' (YouTube/social o diretto) oppure carica un 'file'.")
    if url and file:
        raise HTTPException(400, "Invia o l'URL o il file, non entrambi.")

    tmp_path = None
    cleanup = True
    try:
        # caso URL
        if url:
            tmp_path, source = download_video(url, settings.TMP_DIR, settings.MAX_UPLOAD_MB)
        else:
            # caso file
            if not file.filename.lower().endswith(settings.ALLOWED_EXTS):
                raise HTTPException(400, "Estensione non valida. Usa mp4/mov/mkv/webm/avi.")
            # limite dimensione 50MB
            contents = await file.read()
            if len(contents) > settings.MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(413, f"File troppo grande: max {settings.MAX_UPLOAD_MB}MB.")
            tmp_path = os.path.join(settings.TMP_DIR, file.filename)
            os.makedirs(settings.TMP_DIR, exist_ok=True)
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
        if tmp_path and cleanup:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
