# social_context.py
import os, re, requests, datetime as dt
from urllib.parse import urlparse, parse_qs

YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "").strip()

YT_VIDEO_RE = re.compile(r"(?:v=|/shorts/|/live/|/embed/|youtu\.be/)([A-Za-z0-9_\-]{6,})", re.I)

def is_youtube_url(url:str)->bool:
    host = (urlparse(url).hostname or "").lower() if url else ""
    return any(h in (host or "") for h in ["youtube.com","youtu.be"])

def extract_youtube_id(url:str)->str|None:
    if not url: return None
    q = parse_qs(urlparse(url).query).get("v", [])
    if q: return q[0]
    m = YT_VIDEO_RE.search(url)
    return m.group(1) if m else None

def yt_get(path:str, params:dict):
    if not YOUTUBE_API_KEY:
        raise RuntimeError("YouTube API key missing (set YOUTUBE_API_KEY)")
    base = "https://www.googleapis.com/youtube/v3/"+path.lstrip("/")
    p = dict(params or {})
    p["key"] = YOUTUBE_API_KEY
    r = requests.get(base, params=p, timeout=10)
    r.raise_for_status()
    return r.json()

def youtube_context_score(video_id:str)->dict:
    # 1) Video info
    vi = yt_get("videos", {"part":"snippet,contentDetails,statistics,status","id":video_id})
    items = vi.get("items", [])
    if not items:
        return {"score":0.3, "signals":["Video not found"], "platform":"youtube", "video_id":video_id}
    v = items[0]
    snip = v.get("snippet", {}) or {}
    stats = v.get("statistics", {}) or {}
    status = v.get("status", {}) or {}

    channel_id = snip.get("channelId","")
    title = (snip.get("title") or "").strip()
    desc = (snip.get("description") or "").strip()
    published = snip.get("publishedAt")

    # 2) Channel info
    ci = yt_get("channels", {"part":"snippet,statistics,brandingSettings","id":channel_id})
    citems = ci.get("items", [])
    c = citems[0] if citems else {}
    cs = c.get("statistics", {}) or {}
    c_sn = c.get("snippet", {}) or {}

    # Heuristics → score in [0..1]
    score = 0.5
    signals = []

    # Channel age & subs
    c_published = c_sn.get("publishedAt")
    if c_published:
        try:
            age_years = (dt.datetime.utcnow() - dt.datetime.fromisoformat(c_published.replace("Z","+00:00"))).days/365.25
            if age_years >= 3: score += 0.1; signals.append("Old channel")
            elif age_years < 0.2: score -= 0.1; signals.append("Very new channel")
        except Exception:
            pass

    subs = int(cs.get("subscriberCount","0") or 0)
    if subs >= 100000: score += 0.08; signals.append("High subscriber count")
    elif subs < 50: score -= 0.08; signals.append("Very low subscriber count")

    views = int(stats.get("viewCount","0") or 0)
    likecount = int(stats.get("likeCount","0") or 0)
    if views >= 100000 and likecount >= 1000: score += 0.05; signals.append("Healthy engagement")
    if "made with ai" in (title+desc).lower() or "ai generated" in (title+desc).lower():
        score += 0.02; signals.append("Self-declared AI mention")

    # Title/desc sanity (over-claims / clickbait-y)
    clickbait_terms = ["shocking","you won’t believe","leaked","impossible","100% real"]
    if any(t in (title+desc).lower() for t in clickbait_terms):
        score -= 0.05; signals.append("Clickbait phrasing")

    # Monetization/embeddable/public
    if status.get("embeddable", True): score += 0.01
    if status.get("privacyStatus") != "public": score -= 0.05; signals.append("Not public")

    # Clamp
    score = max(0.0, min(1.0, score))

    # Map to context label
    if score >= 0.70:
        ctx_label = "Context suggests authentic"
    elif score < 0.40:
        ctx_label = "Context suspicious"
    else:
        ctx_label = "Context inconclusive"

    return {
        "score": round(score,4),
        "signals": signals,
        "platform": "youtube",
        "video_id": video_id,
        "context_label": ctx_label
    }
