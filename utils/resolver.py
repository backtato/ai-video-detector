import os
import re
import tempfile
import requests
from typing import Optional
from ..config import RESOLVER_MAX_BYTES, DOWNLOAD_TIMEOUT, DEFAULT_UA

HLS_PAT = re.compile(r"\.m3u8(\?|$)", re.IGNORECASE)

def _headers():
    return {"User-Agent": DEFAULT_UA, "Accept": "*/*"}

def is_hls(url: str, content_type: Optional[str]) -> bool:
    if HLS_PAT.search(url or ""):
        return True
    if content_type and "application/vnd.apple.mpegurl" in content_type.lower():
        return True
    if content_type and "application/x-mpegURL" in content_type:
        return True
    return False

def fetch_to_temp(url: str) -> str:
    """
    Download up to RESOLVER_MAX_BYTES to a temp file.
    For HLS, try to download the master playlist or a short clip via ffmpeg (future).
    """
    with requests.get(url, headers=_headers(), stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        # For now: we still download bytes (some CDNs serve mp4 even with query params)
        total = 0
        fd, tmp = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        try:
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1024 * 64):
                    if not chunk:
                        continue
                    f.write(chunk)
                    total += len(chunk)
                    if total > RESOLVER_MAX_BYTES:
                        break
            return tmp
        except Exception:
            try:
                os.remove(tmp)
            except Exception:
                pass
            raise
