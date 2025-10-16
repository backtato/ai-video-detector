
import os
import shutil
import uuid
import requests
from urllib.parse import urlparse
from typing import Optional, Tuple
from app.config import settings

SOCIAL_DOMAINS = [
    "youtube.com", "youtu.be", "m.youtube.com",
    "tiktok.com", "vm.tiktok.com",
    "twitter.com", "x.com",
    "instagram.com", "www.instagram.com",
    "facebook.com", "fb.watch",
    "reddit.com", "v.redd.it",
]

def _is_social(url: str) -> bool:
    netloc = urlparse(url).netloc.lower()
    return any(d in netloc for d in SOCIAL_DOMAINS)

def _safe_ext(path: str) -> str:
    for ext in settings.ALLOWED_EXTS:
        if path.lower().endswith(ext):
            return ext
    return ".mp4"

def _download_direct(url: str, dest_dir: str, size_limit_bytes: int) -> str:
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    total = 0
    fname = os.path.basename(urlparse(url).path) or f"video-{uuid.uuid4().hex}.mp4"
    ext = _safe_ext(fname)
    out_path = os.path.join(dest_dir, f"dl-{uuid.uuid4().hex}{ext}")
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*128):
            if not chunk:
                continue
            total += len(chunk)
            if total > size_limit_bytes:
                try:
                    os.remove(out_path)
                except Exception:
                    pass
                raise ValueError("Il file supera il limite di 50MB.")
            f.write(chunk)
    return out_path

def _download_yt_dlp(url: str, dest_dir: str, size_limit_bytes: int) -> str:
    from yt_dlp import YoutubeDL
    out_path = os.path.join(dest_dir, f"dl-{uuid.uuid4().hex}.%(ext)s")
    ydl_opts = {
        "outtmpl": out_path,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "format": "best[height<=360][filesize<=50M]/best[height<=360]/best[filesize<=50M]/worst",
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        real_path = ydl.prepare_filename(info)
    if os.path.getsize(real_path) > size_limit_bytes:
        try:
            os.remove(real_path)
        except Exception:
            pass
        raise ValueError("Il video scaricato supera il limite di 50MB. Prova un link più corto o qualità più bassa.")
    ext = _safe_ext(real_path)
    if not real_path.lower().endswith(ext):
        new_path = os.path.splitext(real_path)[0] + ext
        shutil.move(real_path, new_path)
        real_path = new_path
    return real_path

def download_video(url: str, dest_dir: Optional[str] = None, max_mb: int = 50) -> Tuple[str, str]:
    if not dest_dir:
        dest_dir = settings.TMP_DIR
    os.makedirs(dest_dir, exist_ok=True)
    size_limit_bytes = max_mb * 1024 * 1024

    if _is_social(url):
        path = _download_yt_dlp(url, dest_dir, size_limit_bytes)
        return path, "yt-dlp"
    else:
        path = _download_direct(url, dest_dir, size_limit_bytes)
        return path, "direct"
