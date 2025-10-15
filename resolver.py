import os, re, tempfile, requests, subprocess
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

DEFAULT_MAX_BYTES = int(os.environ.get("RESOLVER_MAX_BYTES", "52428800"))  # 50 MB
ALLOWED_DOMAINS = [d.strip().lower() for d in os.environ.get("RESOLVER_ALLOWLIST", "").split(",") if d.strip()]

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"

def is_domain_allowed(url: str) -> bool:
    if not ALLOWED_DOMAINS:
        return True
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)

def fetch_head(url: str, timeout=8):
    try:
        return requests.head(url, allow_redirects=True, timeout=timeout, headers={"User-Agent": UA})
    except Exception:
        return None

def is_video_content(resp) -> bool:
    if not resp:
        return False
    ctype = (resp.headers or {}).get("Content-Type", "")
    if ctype.startswith("video/"):
        return True
    disp = (resp.headers or {}).get("Content-Disposition", "")
    return bool(re.search(r'\.mp4|\.webm|\.mov', resp.url, re.I) or re.search(r'\.mp4|\.webm|\.mov', disp, re.I))

def guess_suffix_from_headers(resp) -> str:
    ctype = (resp.headers or {}).get("Content-Type", "")
    if "mp4" in ctype: return ".mp4"
    if "webm" in ctype: return ".webm"
    if "quicktime" in ctype or "mov" in ctype: return ".mov"
    if re.search(r"\.mp4(\?.*)?$", resp.url, re.I): return ".mp4"
    if re.search(r"\.webm(\?.*)?$", resp.url, re.I): return ".webm"
    if re.search(r"\.mov(\?.*)?$", resp.url, re.I): return ".mov"
    return ".bin"

def download_to_temp(url: str, max_bytes: int = DEFAULT_MAX_BYTES, timeout=20, suffix=None):
    with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": UA}) as r:
        r.raise_for_status()
        if suffix is None:
            suffix = guess_suffix_from_headers(r)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        total = 0
        for chunk in r.iter_content(chunk_size=65536):
            if not chunk: continue
            total += len(chunk)
            if total > max_bytes:
                tmp.close(); os.unlink(tmp.name)
                raise ValueError("File exceeds max size")
            tmp.write(chunk)
        tmp.flush(); tmp.close()
        return tmp.name

def resolve_video_url_from_html(url: str, timeout=10):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": UA})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for key in ["og:video", "og:video:url", "og:video:secure_url"]:
        og = soup.find("meta", attrs={"property": key})
        if og and og.get("content"):
            return urljoin(url, og["content"])
    v = soup.find("video")
    if v and v.get("src"):
        return urljoin(url, v.get("src"))
    s = soup.find("source")
    if s and s.get("src"):
        return urljoin(url, s.get("src"))
    return None

def handle_hls_to_temp(m3u8_url: str, max_seconds: int = 8):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = tmp.name; tmp.close()
    cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error","-i",m3u8_url,"-t",str(max_seconds),"-c","copy",tmp_path]
    try:
        subprocess.check_call(cmd, timeout=60)
    except subprocess.CalledProcessError:
        cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error","-i",m3u8_url,"-t",str(max_seconds),
               "-c:v","libx264","-preset","veryfast","-c:a","aac",tmp_path]
        subprocess.check_call(cmd, timeout=90)
    if os.path.getsize(tmp_path) > DEFAULT_MAX_BYTES:
        os.unlink(tmp_path)
        raise ValueError("HLS sample exceeds max size")
    return tmp_path

def resolve_to_tempfile(url: str):
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("Only http/https supported")
    if not is_domain_allowed(url):
        raise ValueError("Domain not allowed")

    if re.search(r"\.m3u8(\?.*)?$", url, re.I):
        return handle_hls_to_temp(url, max_seconds=8)
    if re.search(r"\.mpd(\?.*)?$", url, re.I):
        raise ValueError("DASH (.mpd) not supported in MVP")

    h = fetch_head(url)
    if h and is_video_content(h):
        return download_to_temp(h.url, suffix=guess_suffix_from_headers(h))

    try:
        vurl = resolve_video_url_from_html(url)
        if vurl:
            if re.search(r"\.m3u8(\?.*)?$", vurl, re.I):
                return handle_hls_to_temp(vurl, max_seconds=8)
            hh = fetch_head(vurl)
            if hh and is_video_content(hh):
                return download_to_temp(hh.url, suffix=guess_suffix_from_headers(hh))
            if re.search(r"\.(mp4|webm|mov)(\?.*)?$", vurl, re.I):
                return download_to_temp(vurl)
    except Exception:
        pass

    if re.search(r"\.(mp4|webm|mov)(\?.*)?$", url, re.I):
        return download_to_temp(url)

    raise ValueError("Unable to resolve a direct video URL")
