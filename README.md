# AI Video Detector (Resolver) – Backend (EN)

## Endpoints
- `GET /` , `GET /healthz`
- `POST /analyze` (file upload)
- `POST /analyze-url` (direct media / HLS sample)
- `POST /analyze-link` (**context-only**; MVP supports YouTube links)

## Response (content analysis)
- `ai_plausibility` [0..1], `confidence` [0..1], `label`: "Likely AI" | "Likely Original" | "Inconclusive"

## Response (context analysis)
- `context_only`: true
- `platform`: "youtube"
- `context_trust_score` [0..1]
- `context_label`: "Context suggests authentic" | "Context suspicious" | "Context inconclusive"
- `context_signals`: [ ... ]
- `label`: "Not assessable (content unavailable)"  (content verdict unavailable; use screen capture or file upload for a pixel-based verdict)

## Environment
- `YOUTUBE_API_KEY` → required for `/analyze-link` (YouTube)
- `RESOLVER_ALLOWLIST`, `RESOLVER_MAX_BYTES`

## Deploy
Render Web Service (Docker). Set Health Check Path `/healthz`.
