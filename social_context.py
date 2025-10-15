import re
import requests
from datetime import datetime, timezone

def is_youtube_url(url: str) -> bool:
    u = url.lower()
    return ("youtube.com" in u) or ("youtu.be" in u)

def extract_youtube_id(url: str) -> str | None:
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", url)
    if m: return m.group(1)
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", url)
    if m: return m.group(1)
    m = re.search(r"youtube\.com/shorts/([A-Za-z0-9_-]{11})", url)
    if m: return m.group(1)
    return None

def _years_since(iso_str: str) -> float:
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z","+00:00"))
        return max(0.0, (datetime.now(timezone.utc)-dt).days/365.25)
    except Exception:
        return 0.0

def _safe_json(resp: requests.Response) -> tuple[dict, str | None]:
    """
    Ritorna (json_dict, error_string). Se la decodifica fallisce o status != 200, l'errore è valorizzato.
    """
    try:
        data = resp.json()
    except Exception:
        text = (resp.text or "")[:300]
        return {}, f"Non-JSON response (status {resp.status_code}): {text}"
    if resp.status_code != 200:
        # YouTube può rispondere 4xx con JSON: surfaciamoselo
        err_msg = data.get("error", {}).get("message") if isinstance(data, dict) else None
        return data if isinstance(data, dict) else {}, err_msg or f"HTTP {resp.status_code}"
    return data, None

def youtube_context_score(video_id: str, api_key: str) -> dict:
    """
    Restituisce:
      - score: 0..100
      - label: "Context suggests authentic" | "Context suspicious" | "Context inconclusive"
      - signals: list
    In caso di problemi di API, solleva ValueError con un messaggio chiaro.
    """
    vresp = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={"part":"snippet,contentDetails,statistics,status","id":video_id,"key":api_key},
        timeout=10
    )
    vi, err = _safe_json(vresp)
    if err:
        raise ValueError(f"YouTube videos API error: {err}")

    items = vi.get("items", []) if isinstance(vi, dict) else []
    if not items:
        # Potrebbe essere video rimosso/privato o id invalido
        return {"score": 50, "label":"Context inconclusive", "signals": ["Video not found or private"]}

    v = items[0]
    snip = v.get("snippet", {}) if isinstance(v, dict) else {}
    ch_id = snip.get("channelId")
    title = (snip.get("title") or "").lower()
    description = (snip.get("description") or "").lower()

    subs = 0
    ch_age_years = 0.0
    if ch_id:
        cresp = requests.get(
            "https://www.googleapis.com/youtube/v3/channels",
            params={"part":"snippet,statistics,status","id":ch_id,"key":api_key},
            timeout=10
        )
        ch, cerr = _safe_json(cresp)
        if cerr:
            # Non è “fatale”, ma abbassiamo la fiducia e segnaliamo
            ch = {}
            ch_err_signal = f"Channel API error: {cerr}"
        else:
            ch_err_signal = None

        chi = ch.get("items") or [] if isinstance(ch, dict) else []
        if chi:
            c = chi[0]
            try:
                subs = int(c.get("statistics", {}).get("subscriberCount", 0) or 0)
            except Exception:
                subs = 0
            ch_age_years = _years_since(c.get("snippet", {}).get("publishedAt",""))
    else:
        ch_err_signal = "Missing channelId in snippet"

    # Heuristics → score 0..100
    score = 50
    signals = []

    if subs > 1_000_000:
        score += 10; signals.append("Large audience channel")
    elif subs > 100_000:
        score += 6; signals.append("Significant audience channel")
    elif subs < 1_000:
        score -= 5; signals.append("Very small audience")

    if ch_age_years >= 5:
        score += 6; signals.append(f"Old channel (~{ch_age_years:.1f}y)")
    elif ch_age_years < 0.5:
        score -= 6; signals.append("Very new channel")

    clickbait_terms = ["shocking", "you won't believe", "gone wrong", "impossible", "ai generated"]
    if any(t in title for t in clickbait_terms):
        score -= 5; signals.append("Clickbait-like title")
    if " ai " in f" {title} " or " ai " in f" {description} ":
        signals.append("Mentions AI in title/description")

    if ch_err_signal:
        signals.append(ch_err_signal)

    score = max(0, min(100, score))
    if score >= 65:
        label = "Context suggests authentic"
    elif score <= 35:
        label = "Context suspicious"
    else:
        label = "Context inconclusive"

    return {"score": score, "label": label, "signals": signals}
