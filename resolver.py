import os
import re
import socket
import urllib.parse
import tempfile
import requests
from contextlib import contextmanager
from typing import Optional

from config import RESOLVER_ALLOWLIST, RESOLVER_MAX_BYTES, FFMPEG_USER_AGENT, FFMPEG_RW_TIMEOUT_US, HLS_SAMPLE_SECONDS

PRIVATE_NETS = (
    ("10.",),
    ("172.", "16."),  # coarse check; sicurezza sufficiente per MVP
    ("192.168.",),
    ("127.",),
)

def _is_private_ip(host: str) -> bool:
    try:
        ip = socket.gethostbyname(host)
    except socket.gaierror:
        return True  # host non risolvibile: blocca
    if ip == "::1":
        return True
    return any(ip.startswith(prefix) for group in PRIVATE_NETS for prefix in group)

if RESOLVER_ALLOWLIST.strip():
    _ALLOW = set(h.strip().lower() for h in RESOLVER_ALLOWLIST.split(",") if h.strip())
    def is_allowed(url: str) -> bool:
        u = urllib.parse.urlparse(url)
        return u.scheme in ("http", "https") and (u.hostname or "").lower() in _ALLOW
else:
    # default permissivo ma sicuro
    def is_allowed(url: str) -> bool:
        u = urllib.parse.urlparse(url)
        host = (u.hostname or "").lower()
        if u.scheme not in ("http", "https") or not host:
            return False
        return not _is_private_ip(host)

def _cap_stream_write(resp: requests.Response, tmp, cap_bytes: int) -> int:
    total = 0
    for chunk in resp.iter_content(chunk_size=65536):
        if not chunk:
            continue
        total += len(chunk)
        if total > cap_bytes:
            raise ValueError("MAX_BYTES_EXCEEDED")
        tmp.write(chunk)
    return total

@contextmanager
def download_to_temp(url: str, cap_bytes: int):
    if not is_allowed(url):
        raise PermissionError("URL not allowed by policy")
    headers = {"User-Agent": "Mozilla/5.0 (AI-Video/1.0)"}
    with requests.get(url, headers=headers, stream=True, timeout=(10, 30)) as r:
        r.raise_for_status()
        suffix = os.path.splitext(urllib.parse.urlparse(url).path)[1] or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            try:
                _cap_stream_write(r, tmp, cap_bytes)
                path = tmp.name
            except Exception:
                tmp.close()
                try: os.unlink(tmp.name)
                except Exception: pass
                raise
    try:
        yield path
    finally:
        try: os.unlink(path)
        except Exception: pass

def sample_hls_to_file(m3u8_url: str, seconds: int = HLS_SAMPLE_SECONDS) -> str:
    # Estrae ~N secondi con ffmpeg, user-agent e timeout
    import subprocess, tempfile
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); out.close()
    cmd = [
        "ffmpeg",
        "-user_agent", FFMPEG_USER_AGENT,
        "-rw_timeout", str(FFMPEG_RW_TIMEOUT_US),
        "-y",
        "-i", m3u8_url,
        "-t", str(seconds),
        "-c", "copy",
        out.name
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(out.name) or os.path.getsize(out.name) == 0:
        try: os.unlink(out.name)
        except Exception: pass
        raise RuntimeError(f"ffmpeg HLS sample failed: {p.stderr[-400:]}")
    return out.name
