**# AI Video Detector (Resolver) – Backend
Deploy on Render/ Railway / Fly. Env vars:
- RESOLVER_ALLOWLIST: comma-separated domains (e.g., "yourdomain.com,cdn.example.com")
- RESOLVER_MAX_BYTES: max download size in bytes (default 50 MB)
Endpoints:
- POST /analyze (multipart file)
- POST /analyze-url (json {"url": "..."})
**
