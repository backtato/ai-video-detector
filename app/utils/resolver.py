# utils/resolver.py
import os
import re
import tempfile
import asyncio
import subprocess
from typing import Optional

import yt_dlp
from config import DOWNLOAD_TIMEOUT, FFMPEG_TRIM_SECONDS, TMP_DIR

YOUTUBE_RX = re.compile(r"(youtube\.com|youtu\.be)", re.I)
HLS_RX = re.compile(r"\.m3u8($|\?)", re.I)


def _ffmpeg_trim(src: str, dst: str, seconds: int = 20) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-t", str(seconds),
        "-c", "copy",
        dst,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


async def resolve_to_media_file(url: str, max_bytes: int) -> Optional[str]:
    """
    Ritorna il path a un file video locale (primi ~20s).
    - Se URL è YouTube/Shorts → yt-dlp scarica il best video+audio e poi tronca a 20s.
    - Se URL è un HLS (.m3u8) → usa ffmpeg per salvare 20s locali.
    - Altri URL diretti a .mp4/.mov/etc → yt-dlp fallback per maggiore compatibilità.
    """
    tmp_raw = tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR, suffix=".mp4")
    tmp_raw.close()
    raw_path = tmp_raw.name

    tmp_out = tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR, suffix=".mp4")
    tmp_out.close()
    out_path = tmp_out.name

    # yt-dlp options conservative
    ydl_opts = {
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "geo_bypass": True,
        "retries": 1,
        "outtmpl": raw_path,
        "overwrites": True,
        # preferisci mp4
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        # limite dimensione indicativo: se superato, yt-dlp abortisce
        "file_access_retries": 1,
        "socket_timeout": 15,
    }

    async def _download_with_ytdlp() -> bool:
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return True
        except Exception:
            return False

    async def _download_hls_ffmpeg() -> bool:
        # 20s HLS
        cmd = [
            "ffmpeg", "-y",
            "-stimeout", "15000000",   # 15s
            "-i", url,
            "-t", str(FFMPEG_TRIM_SECONDS),
            "-c", "copy",
            raw_path,
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            try:
                await asyncio.wait_for(proc.communicate(), timeout=DOWNLOAD_TIMEOUT)
            except asyncio.TimeoutError:
                proc.kill()
                return False
            return proc.returncode == 0
        except Exception:
            return False

    # Strategy
    ok = False
    if HLS_RX.search(url):
        ok = await _download_hls_ffmpeg()
    else:
        ok = await _download_with_ytdlp()

    if not ok or not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
        # Fallback tentativo HLS se sembrava non-HLS ma yt-dlp non ce l’ha fatta
        if not HLS_RX.search(url):
            hls_try = await _download_hls_ffmpeg()
            if not hls_try:
                try:
                    os.remove(raw_path)
                except Exception:
                    pass
                return None
        else:
            try:
                os.remove(raw_path)
            except Exception:
                pass
            return None

    # Tronca ai primi N secondi (stream copy, veloce)
    trimmed = _ffmpeg_trim(raw_path, out_path, seconds=FFMPEG_TRIM_SECONDS)
    try:
        os.remove(raw_path)
    except Exception:
        pass

    if not trimmed:
        return None

    # Byte-guard
    if os.path.getsize(out_path) > max_bytes:
        try:
            os.remove(out_path)
        except Exception:
            pass
        return None

    return out_path
