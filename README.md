# AI Video Detector (Resolver) – Backend (EN)

**What it does**
- Analyzes video content to estimate if it is AI-generated (heuristic MVP).
- Supports file uploads and direct URLs. For HLS `.m3u8`, it samples ~8s with `ffmpeg`.

**Endpoints**
- `GET /`          → service status
- `GET /healthz`   → health check
- `POST /analyze`  → multipart file upload
- `POST /analyze-url` → `{ "url": "https://..." }` (direct video or resolvable HLS)

**Response**
- `ai_plausibility` [0..1]
- `confidence` [0..1]
- `label` → `Likely AI` | `Likely Original` | `Inconclusive`
- `explanations`, `video_info`

**Environment**
- `RESOLVER_ALLOWLIST` → comma-separated allowlist (e.g., `yourdomain.com,cdn.example.com`)
- `RESOLVER_MAX_BYTES` → max download size in bytes (default 50 MB)

**Deploy on Render**
- Create a Web Service from this repo (Docker).
- Set **Health Check Path** to `/healthz`.
- Free tier may cold-start; consider a retry on the client (the provided WP plugin handles a single 502 retry).
