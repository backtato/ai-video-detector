# AI Video Detector (CPU-only) — URL Social/YouTube o Upload (<= 50MB)

## Endpoints
- `GET /` → form HTML minimale
- `GET /healthz`
- `POST /predict` (form-data: `url` *oppure* `file`)
- `GET /predict-get?url=...`
- `GET /predict?url=...` (alias)

## Esempi
curl -F "url=https://www.youtube.com/watch?v=U2BEMKvyT4U" https://<host>/predict
open "https://<host>/predict?url=https://www.youtube.com/watch?v=U2BEMKvyT4U"