import re
import requests
from datetime import datetime, timezone

def is_youtube_url(url: str) -> bool:
    u = url.lower()
    return ("youtube.com" in u) or ("youtu.be" in u)

def extract_youtube_id(url: str) -> str | None:
    # ?v=ID
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", url)
    if m: return m.group(1)
    # youtu.be/ID
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", url)
    if m: return m.group(1)
    # youtube.com/shorts/ID
    m = re.search(r"youtube\.com/shorts/([A-Za-z0-9_-]{11})", url)
    if m: return m.group(1)
    return None

def _years_since(iso_str: str) -> float:
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z","+00:00"))
        return max(0.0, (datetime.now(timezone.utc)-dt).days/365.25)
    except Exception:
        return 0.0

def youtube_context_score(video_id: str, api_key: str) -> dict:
    """
    Restituisce un dizionario:
      - score: 0..100
      - label: "Context suggests authentic" | "Context suspicious" | "Context inconclusive"
      - signals: elenco segnali
    """
    # Video details
    vresp = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={"part":"snippet,contentDetails,statistics,status","id":video_id,"key":api_key},
        timeout=10
    )
    vi = vresp.json()
    items = vi.get("items", [])
    if not items:
        return {"score": 50, "label":"Context inconclusive", "signals": ["Video not found or private"]}

    v = items[0]
    snip = v.get("snippet", {})
    ch_id = snip.get("channelId")
    title = (snip.get("title") or "").lower()
    description = (snip.get("description") or "").lower()

    # Channel
    subs = 0
    ch_age_years = 0.0
    if ch_id:
        cresp = requests.get(
            "https://www.googleapis.com/youtube/v3/channels",
            params={"part":"snippet,statistics,status","id":ch_id,"key":api_key},
            timeout=10
        )
        ch = cresp.json()
        chi = ch.get("items") or []
        if chi:
            c = chi[0]
            subs = int(c.get("statistics", {}).get("subscriberCount", 0) or 0)
            ch_age_years = _years_since(c.get("snippet", {}).get("publishedAt",""))

    # Heuristics → score 0..100
    score = 50
    signals = []

    # audience
    if subs > 1_000_000:
        score += 10; signals.append("Large audience channel")
    elif subs > 100_000:
        score += 6; signals.append("Significant audience channel")
    elif subs < 1_000:
        score -= 5; signals.append("Very small audience")

    # channel age
    if ch_age_years >= 5:
        score += 6; signals.append(f"Old channel (~{ch_age_years:.1f}y)")
    elif ch_age_years < 0.5:
        score -= 6; signals.append("Very new channel")

    # content cues
    clickbait_terms = ["shocking", "you won't believe", "gone wrong", "impossible", "ai generated"]
    if any(t in title for t in clickbait_terms):
        score -= 5; signals.append("Clickbait-like title")
    if " ai " in f" {title} " or " ai " in f" {description} ":
        signals.append("Mentions AI in title/description")

    score = max(0, min(100, score))
    if score >= 65:
        label = "Context suggests authentic"
    elif score <= 35:
        label = "Context suspicious"
    else:
        label = "Context inconclusive"

    return {"score": score, "label": label, "signals": signals}
