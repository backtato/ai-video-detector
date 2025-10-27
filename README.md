# AI-Video Detector (FastAPI + Gunicorn)
**Author:** Backtato

- Limite upload configurabile via `MAX_UPLOAD_BYTES` (default **100 MB**).
- Localizzazione automatica via `Accept-Language` / `X-Lang` / `?lang=`.

Backend per stimare se un video è **reale / parzialmente AI / AI**, con output:
- `ai_score` (0=molto reale … 1=molto AI)
- `label` (Con alta probabilità è REALE / Esito incerto / Con alta probabilità è AI)
- `timeline` con segmenti sospetti
- metadati (`width/height/fps/duration/bit_rate/codec`) e piccoli check forensi

## Endpoints

- `GET /healthz` → `"ok"`
- `POST /analyze` → upload file (`form-data: file=@video.*`)
- `POST /analyze-url` → analizza link diretto a file video (`form-data: url=https://...`)
  - blocca HLS `.m3u8` e HTML/login page
  - social protetti (Instagram/TikTok ecc.): usa `USE_YTDLP=1` **con cookie**, oppure carica un file/registrazione schermo
- `POST /predict` → retro-compatibile: accetta **`file` o `url`**

## Esempi `curl`

Upload:
```bash
curl -sS -X POST "$BASE/analyze" -F "file=@/path/IMG_4568.mov" | jq
