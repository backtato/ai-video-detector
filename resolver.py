import os, re, tempfile, requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

DEFAULT_MAX_BYTES = int(os.environ.get("RESOLVER_MAX_BYTES", "52428800"))  # 50 MB
ALLOWED_DOMAINS = [d.strip().lower() for d in os.environ.get("RESOLVER_ALLOWLIST", "").split(",") if d.strip()]

def is_domain_allowed(url: str) -> bool:
    if not ALLOWED_DOMAINS:
        return True  # allow all in dev; set allowlist in prod
    host = urlparse(url).hostname or ""
    host = host.lower()
    return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)

def fetch_head(url: str, timeout=8):
    try:
        return requests.head(url, allow_redirects=True, timeout=timeout, headers={"User-Agent": "AIVideoResolver/1.0"})
    except Exception:
        return None

def is_video_content(resp) -> bool:
    if not resp:
        return False
    ctype = (resp.headers or {}).get("Content-Type", "")
    return ctype.startswith("video/")

def download_to_temp(url: str, max_bytes: int = DEFAULT_MAX_BYTES, timeout=20):
    with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "AIVideoResolver/1.0"}) as r:
        r.raise_for_status()
        total = 0
        suffix = guess_suffix(r.headers.get("Content-Type", ""))
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        for chunk in r.iter_content(chunk_size=65536):
            if not chunk: continue
            total += len(chunk)
            if total > max_bytes:
                tmp.close(); os.unlink(tmp.name)
                raise ValueError("File exceeds max size")
            tmp.write(chunk)
        tmp.flush(); tmp.close()
        return tmp.name

def guess_suffix(content_type: str) -> str:
    if "mp4" in content_type: return ".mp4"
    if "webm" in content_type: return ".webm"
    if "quicktime" in content_type or "mov" in content_type: return ".mov"
    return ".bin"

def resolve_video_url_from_html(url: str, timeout=10):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "AIVideoResolver/1.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    og = soup.find("meta", attrs={"property": "og:video"})
    if og and og.get("content"):
        return urljoin(url, og["content"])
    v = soup.find("video")
    if v and v.get("src"):
        return urljoin(url, v["src"])
    s = soup.find("source")
    if s and s.get("src"):
        return urljoin(url, s["src"])
    return None

def resolve_to_tempfile(url: str):
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("Only http/https supported")
    if not is_domain_allowed(url):
        raise ValueError("Domain not allowed")

    h = fetch_head(url)
    if h and is_video_content(h):
        return download_to_temp(h.url)

    try:
        vurl = resolve_video_url_from_html(url)
        if vurl:
            hh = fetch_head(vurl)
            if hh and is_video_content(hh):
                return download_to_temp(hh.url)
    except Exception:
        pass

    if re.search(r"\.(mp4|webm|mov)(\?.*)?$", url, re.IGNORECASE):
        return download_to_temp(url)

    raise ValueError("Unable to resolve a direct video URL")
