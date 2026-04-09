"""
render_app/app.py
─────────────────
Deployed on Railway / Render. Zero ML dependencies.
Serves the frontend and proxies verify requests to the Kaggle ML backend.
Feed is self-sufficient via RSS fallback when Kaggle is down.
Images via picsum.photos — no API key, no rate limits.

Environment variables:
  KAGGLE_BACKEND_URL = https://xxxx-xx-xx.ngrok-free.app
"""

import os
import re
import random
import logging
import requests
import feedparser
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR  = os.path.dirname(_THIS_DIR)

app = Flask(
    __name__,
    template_folder=os.path.join(_BASE_DIR, "templates"),
    static_folder=os.path.join(_BASE_DIR, "static"),
    static_url_path="/static",
)
CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

KAGGLE_BACKEND_URL = os.getenv("KAGGLE_BACKEND_URL", "").rstrip("/")

GNEWS_RSS_INDIA = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"

FALLBACK_RSS_FEEDS = [
    ("The Hindu",       "https://www.thehindu.com/feeder/default.rss"),
    ("NDTV",            "https://feeds.feedburner.com/ndtvnews-top-stories"),
    ("Indian Express",  "https://indianexpress.com/feed/"),
    ("Times of India",  "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"),
]

# ─── Picsum image seeds per category ─────────────────────────
# Each number is a stable picsum photo ID that fits the category.
# Browse them at: https://picsum.photos/images

_CATEGORY_SEEDS = {
    "Technology": [0, 1, 20, 48, 119, 160, 180, 201],
    "Business":   [3, 6, 26, 43, 57, 76, 103, 137],
    "Sports":     [17, 29, 35, 60, 80, 120, 140, 165],
    "Health":     [9, 21, 42, 55, 91, 116, 134, 177],
    "Politics":   [12, 33, 44, 67, 88, 109, 130, 155],
    "General":    [2, 5, 10, 15, 22, 30, 50, 70],
}


def get_category_image(headline: str) -> str:
    """Returns a stable picsum image URL based on headline category."""
    hl = headline.lower()

    if any(w in hl for w in ["ai", "tech", "app", "software", "apple", "google", "cyber", "data", "space"]):
        cat = "Technology"
    elif any(w in hl for w in ["market", "bank", "stock", "rupee", "economy", "tata", "reliance", "business", "tax"]):
        cat = "Business"
    elif any(w in hl for w in ["cricket", "match", "kohli", "bcci", "olympic", "sports", "football", "team"]):
        cat = "Sports"
    elif any(w in hl for w in ["doctor", "hospital", "virus", "health", "cancer", "study", "science"]):
        cat = "Health"
    elif any(w in hl for w in ["modi", "election", "bjp", "congress", "govt", "minister", "court", "law", "police"]):
        cat = "Politics"
    else:
        cat = "General"

    seed = random.choice(_CATEGORY_SEEDS[cat])
    return f"https://picsum.photos/id/{seed}/600/400"


def _fetch_rss_fallback() -> list:
    """Fetch news directly from RSS feeds — no Kaggle needed."""
    articles = []

    # Try Google News RSS first
    try:
        feed = feedparser.parse(GNEWS_RSS_INDIA)
        for e in feed.entries[:12]:
            hl = e.get("title", "")
            src = ""
            if e.get("source"):
                src = e.source.get("title", "") if hasattr(e.source, "get") else getattr(e.source, "title", "")
            articles.append({
                "headline": hl,
                "source":   src or "Google News",
                "link":     e.get("link", "#"),
                "image":    get_category_image(hl),
                "time":     e.get("published", ""),
            })
    except Exception as exc:
        logger.warning(f"GNews RSS error: {exc}")

    # Top up from publisher feeds if needed
    if len(articles) < 6:
        for source_name, rss_url in FALLBACK_RSS_FEEDS:
            try:
                feed = feedparser.parse(rss_url)
                for e in feed.entries[:3]:
                    hl = e.get("title", "")
                    articles.append({
                        "headline": hl,
                        "source":   source_name,
                        "link":     e.get("link", "#"),
                        "image":    get_category_image(hl),
                        "time":     getattr(e, "published", ""),
                    })
            except Exception as exc:
                logger.warning(f"RSS fallback error ({source_name}): {exc}")

    return articles[:12]


def _kaggle(path: str) -> str:
    return f"{KAGGLE_BACKEND_URL}{path}"


# ─── Pages ────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ─── Health ───────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    gateway = {"status": "ok", "service": "render_gateway"}

    if not KAGGLE_BACKEND_URL:
        return jsonify({**gateway, "kaggle_backend": {"status": "not_configured"}})

    try:
        resp = requests.get(_kaggle("/api/ml-health"), timeout=8)
        return jsonify({**gateway, "kaggle_backend": resp.json()})
    except Exception as exc:
        return jsonify({**gateway, "kaggle_backend": {"status": "unreachable", "error": str(exc)}})


# ─── Verify → proxy to Kaggle ─────────────────────────────────

@app.route("/api/verify", methods=["POST"])
def verify():
    if not KAGGLE_BACKEND_URL:
        return jsonify({"error": "ML backend not configured. Set KAGGLE_BACKEND_URL on Railway."}), 503

    body = request.get_json(silent=True)
    if not body or not body.get("url"):
        return jsonify({"error": "Missing 'url' field"}), 400

    url = body["url"].strip()
    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "Invalid URL"}), 400

    try:
        resp = requests.post(_kaggle("/api/ml-verify"), json={"url": url}, timeout=120)
        return jsonify(resp.json()), resp.status_code
    except requests.exceptions.Timeout:
        return jsonify({"error": "ML backend timed out. Kaggle may be cold-starting."}), 504
    except Exception as exc:
        return jsonify({"error": f"Backend unreachable: {exc}"}), 502


# ─── Feed: Kaggle first, RSS fallback ─────────────────────────

@app.route("/api/feed", methods=["GET"])
def feed():
    # Try Kaggle backend first
    if KAGGLE_BACKEND_URL:
        try:
            resp = requests.get(_kaggle("/api/ml-feed"), timeout=30)
            data = resp.json()
            if data:
                return jsonify(data)
        except Exception as exc:
            logger.warning(f"Kaggle feed unavailable: {exc}")

    # Self-sufficient fallback — works even without Kaggle
    logger.info("Feed: using RSS fallback")
    return jsonify(_fetch_rss_fallback())


# ─── Dashboard → proxy to Kaggle ──────────────────────────────

@app.route("/api/dashboard-data", methods=["GET"])
def dashboard_data():
    if not KAGGLE_BACKEND_URL:
        return jsonify({"success": False, "error": "Backend not configured"})
    try:
        resp = requests.get(_kaggle("/api/ml-dashboard"), timeout=60)
        return jsonify(resp.json())
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})


if __name__ == "__main__":
    app.run(debug=False, port=int(os.getenv("PORT", 5000)))