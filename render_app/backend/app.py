"""
backend/app.py
SachBol AI — Local Flask server (full ML stack)

Run:
  cd sachbol/
  python -m backend.app
Gunicorn (prod):
  gunicorn -w 1 -b 0.0.0.0:5000 backend.app:app
"""

import os
import sys
import re
import json
import logging
import random
import concurrent.futures
from collections import Counter
from urllib.parse import urlparse

import requests
import feedparser
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_THIS_DIR)

if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

app = Flask(
    __name__,
    template_folder=os.path.join(_BASE_DIR, "templates"),
    static_folder=os.path.join(_BASE_DIR, "static"),
    static_url_path="/static",
)
CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                    datefmt="%m/%d/%Y %I:%M:%S %p")
logger = logging.getLogger(__name__)

# ── Feed cache — avoids re-fetching GNews on every page load ──
_feed_cache: dict = {"data": [], "ts": 0.0}
FEED_TTL_SECONDS  = 300   # 5 minutes

# ─── Helpers ──────────────────────────────────────────────────

_NEWS_STOP_WORDS = {
    "the", "and", "for", "that", "with", "from", "this", "news",
    "live", "update", "says", "will", "after", "what", "where",
    "when", "google", "full", "coverage", "india", "over", "into",
}

_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

GNEWS_RSS_INDIA = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"

# ─── Unsplash Category Image Logic ────────────────────────────

CATEGORY_IMAGES = {
    "Technology": [
        "https://images.unsplash.com/photo-1518770660439-4636190af475?w=600&h=400&fit=crop", 
        "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=600&h=400&fit=crop", 
        "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=600&h=400&fit=crop"  
    ],
    "Business": [
        "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=600&h=400&fit=crop", 
        "https://images.unsplash.com/photo-1444653614773-995cb1ef9efa?w=600&h=400&fit=crop", 
        "https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=600&h=400&fit=crop"  
    ],
    "Sports": [
        "https://images.unsplash.com/photo-1461896836934-ffe607ba8211?w=600&h=400&fit=crop", 
        "https://images.unsplash.com/photo-1540747913346-19e32dc3e97e?w=600&h=400&fit=crop", 
        "https://images.unsplash.com/photo-1517649763962-0c623066013b?w=600&h=400&fit=crop"  
    ],
    "Health": [
        "https://images.unsplash.com/photo-1505751172876-fa1923c5c528?w=600&h=400&fit=crop", 
        "https://images.unsplash.com/photo-1530497610245-94d3c16cda28?w=600&h=400&fit=crop", 
        "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=600&h=400&fit=crop"  
    ],
    "Politics": [
        "https://images.unsplash.com/photo-1526470608159-98544929315b?w=600&h=400&fit=crop", 
        "https://images.unsplash.com/photo-1541872703-74c5e44368f9?w=600&h=400&fit=crop",    
        "https://images.unsplash.com/photo-1555848962-6e79363ec58f?w=600&h=400&fit=crop"     
    ],
    "General": [
        "https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=600&h=400&fit=crop", 
        "https://images.unsplash.com/photo-1495020689067-958852a7765e?w=600&h=400&fit=crop", 
        "https://images.unsplash.com/photo-1585829365295-ab7cd400c167?w=600&h=400&fit=crop"  
    ]
}

def get_category_image(headline: str) -> str:
    """Assigns a random curated Unsplash image based on keywords in the headline."""
    headline_lower = headline.lower()
    
    if any(w in headline_lower for w in ["ai", "tech", "app", "software", "apple", "google", "cyber", "data", "space"]):
        category = "Technology"
    elif any(w in headline_lower for w in ["market", "bank", "stock", "rupee", "economy", "tata", "reliance", "business", "tax"]):
        category = "Business"
    elif any(w in headline_lower for w in ["cricket", "match", "kohli", "bcci", "olympic", "sports", "football", "team"]):
        category = "Sports"
    elif any(w in headline_lower for w in ["doctor", "hospital", "virus", "health", "cancer", "study", "science"]):
        category = "Health"
    elif any(w in headline_lower for w in ["modi", "election", "bjp", "congress", "govt", "minister", "court", "law", "police"]):
        category = "Politics"
    else:
        category = "General"
        
    return random.choice(CATEGORY_IMAGES[category])

def get_word_trends(articles: list) -> list:
    """Fallback: word-frequency trends from news headlines."""
    text  = " ".join(a.get("title", "") for a in articles)
    words = re.findall(r"\w+", text.lower())
    words = [w for w in words if w not in _NEWS_STOP_WORDS and len(w) > 4]
    pairs = Counter(words).most_common(10)
    return [{"topic": w, "rank": i+1, "count": c} for i, (w, c) in enumerate(pairs)]

def resolve_google_news_url(url: str) -> str:
    """Follow Google News RSS redirects to reach the real publisher URL for the Verify Agent."""
    if "news.google.com" not in url:
        return url
    try:
        resp = requests.head(
            url,
            headers={"User-Agent": _BROWSER_UA, "Accept-Language": "en-US,en;q=0.9"},
            allow_redirects=True,
            timeout=5,
        )
        final = resp.url
        if final and "google.com" not in final and final != url:
            return final
    except Exception:
        pass
    return url

FALLBACK_RSS_FEEDS = [
    ("The Hindu",        "https://www.thehindu.com/feeder/default.rss"),
    ("NDTV",             "https://feeds.feedburner.com/ndtvnews-top-stories"),
    ("Indian Express",   "https://indianexpress.com/feed/"),
    ("Times of India",   "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"),
]

def _fetch_rss_articles(max_per_feed: int = 3) -> list[dict]:
    """Fetch fallback articles directly from publisher RSS feeds, using Unsplash images."""
    try:
        import feedparser
    except ImportError:
        logger.warning("feedparser not installed — skipping RSS fallback")
        return []

    results = []
    for source_name, rss_url in FALLBACK_RSS_FEEDS:
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:max_per_feed]:
                link = entry.get("link", "#")
                headline_text = entry.get("title", "")
                published = entry.published if hasattr(entry, "published") else ""

                results.append({
                    "headline": headline_text,
                    "source":   source_name,
                    "link":     link,
                    "image":    get_category_image(headline_text),
                    "time":     published,
                    "_resolved_link": link,
                })
        except Exception as exc:
            logger.warning(f"RSS feed error ({source_name}): {exc}")

    return results

# ─── Pages ────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# ─── API: Fact-check (Ensemble) ───────────────────────────────

@app.route("/api/verify", methods=["POST"])
def verify():
    from agent import run_agent

    body = request.get_json(silent=True)
    if not body or not body.get("url"):
        return jsonify({"error": "Missing 'url' field in request body"}), 400

    url = body["url"].strip()
    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "Invalid URL — must start with http:// or https://"}), 400

    hint_title = body.get("title", "").strip()
    result = run_agent(url, hint_title=hint_title)

    if "error" in result:
        return jsonify({"error": result["error"]}), 500

    return jsonify(result)

# ─── API: News feed ───────────────────────────────────────────

@app.route("/api/feed", methods=["GET"])
def feed():
    """Returns top Indian news headlines with instantly loaded categorized images."""
    import time

    if time.time() - _feed_cache["ts"] < FEED_TTL_SECONDS and _feed_cache["data"]:
        logger.info("Feed: serving from cache")
        return jsonify(_feed_cache["data"])

    articles_out = []
    try:
        feed_data = feedparser.parse(GNEWS_RSS_INDIA)
        entries   = feed_data.entries[:14]

        if entries:
            # Resolve Google News redirect URLs concurrently for the Verify agent
            raw_links = [e.get("link", "#") for e in entries]
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
                resolved_links = list(pool.map(resolve_google_news_url, raw_links))

            for i, e in enumerate(entries):
                headline = e.get("title", "")
                src = ""
                if e.get("source"):
                    src = e.source.get("title", "") if hasattr(e.source, "get") else getattr(e.source, "title", "")
                
                articles_out.append({
                    "headline": headline,
                    "source":   src or "Google News",
                    "link":     resolved_links[i] or raw_links[i],
                    "image":    get_category_image(headline),
                    "time":     e.get("published", ""),
                })

    except Exception as exc:
        logger.error(f"GNews RSS feed error: {exc}")

    # Fallback to direct publisher RSS feeds if Google News is down
    if len(articles_out) < 6:
        logger.info("Topping up feed with direct publisher RSS feeds")
        articles_out.extend(_fetch_rss_articles(max_per_feed=3))

    result = articles_out[:12]

    if result:
        _feed_cache["data"] = result
        _feed_cache["ts"]   = time.time()

    return jsonify(result)

# ─── API: Dashboard analytics ─────────────────────────────────

_TRENDING_RSS_FEEDS = [
    ("https://feeds.feedburner.com/ndtvnews-top-stories",         "India"),
    ("https://timesofindia.indiatimes.com/rssfeedstopstories.cms","India"),
    ("https://www.thehindu.com/feeder/default.rss",               "India"),
    ("https://indianexpress.com/feed/",                           "India"),
    ("https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml","India"),
    ("https://www.livemint.com/rss/news",                         "Business"),
    ("https://economictimes.indiatimes.com/rssfeedstopstories.cms","Business"),
    ("https://feeds.bbci.co.uk/news/world/rss.xml",               "World"),
    ("https://feeds.reuters.com/reuters/topNews",                 "World"),
    ("https://feeds.bbci.co.uk/news/technology/rss.xml",          "Tech"),
    ("https://feeds.bbci.co.uk/news/science_and_environment/rss.xml","Science"),
    ("https://feeds.bbci.co.uk/news/health/rss.xml",              "Health"),
    ("https://timesofindia.indiatimes.com/rssfeeds/913168846.cms","Sports"),
]

_TREND_STOP = {
    "the","and","for","that","with","from","this","news","live","update",
    "says","will","after","what","where","when","google","full","coverage",
    "india","over","into","have","been","were","their","they","said","also",
    "more","about","would","could","which","there","its","his","her","but",
    "new","year","day","week","time","first","last","how","why","who",
    "after","back","just","make","take","know","than","then","only","even",
    "most","some","your","our","all","one","two","three","not","has","had",
    "can","may","now","yet","top","big","key","amid","hold","call","amid",
    "report","reports","latest","major","gets","goes","meet","plan","says",
}

def _get_rss_trending(max_topics: int = 20) -> list:
    try:
        import feedparser
        import re as _re
    except ImportError:
        return []

    all_titles = []

    def _fetch(url_region):
        url, region = url_region
        try:
            feed = feedparser.parse(url)
            return [(e.get("title", ""), region) for e in feed.entries[:15]]
        except Exception:
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_fetch, _TRENDING_RSS_FEEDS))

    for batch in results:
        all_titles.extend(batch)

    entity_counter: dict = {}
    entity_region:  dict = {}

    for title, region in all_titles:
        entities = _re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', title)
        for ent in entities:
            words = ent.split()
            if len(ent) < 4: continue
            if len(words) == 1 and ent.lower() in _TREND_STOP | {"india","us","uk","un","eu","pm","cm","mr","ms","dr","he","she"}:
                continue
            entity_counter[ent] = entity_counter.get(ent, 0) + 1
            if ent not in entity_region:
                entity_region[ent] = region

    word_counter: dict = {}
    import re as _re2
    for title, region in all_titles:
        words = _re2.findall(r'\b[a-z]{5,}\b', title.lower())
        for w in words:
            if w not in _TREND_STOP:
                word_counter[w] = word_counter.get(w, 0) + 1

    combined: dict = {}
    for ent, cnt in entity_counter.items():
        if cnt >= 2: combined[ent] = cnt * 3.0

    for w, cnt in sorted(word_counter.items(), key=lambda x: -x[1]):
        if w.capitalize() not in combined and cnt >= 3:
            combined[w.capitalize()] = cnt * 1.0

    top = sorted(combined.items(), key=lambda x: -x[1])[:max_topics]
    return [
        {"topic": item[0], "rank": i+1,
         "region": entity_region.get(item[0], "Global"), "count": int(item[1])}
        for i, item in enumerate(top)
    ]

@app.route("/api/dashboard-data", methods=["GET"])
def dashboard_data():
    GNEWS_TOPICS = {
        "Technology":    "TECHNOLOGY",
        "Science":       "SCIENCE",
        "Health":        "HEALTH",
        "Business":      "BUSINESS",
        "Entertainment": "ENTERTAINMENT",
        "Sports":        "SPORTS",
        "Nation":        "NATION",
    }

    cat_counts:  list  = []
    cat_labels:  list  = list(GNEWS_TOPICS.keys())
    all_articles: list = []
    sentiments = {"Positive": 0, "Neutral": 0, "Negative": 0}

    try:
        from gnews import GNews
        from textblob import TextBlob

        gn = GNews(language="en", country="IN", max_results=15)

        def _fetch_cat(topic_pair):
            label, topic_str = topic_pair
            try:
                return label, gn.get_news_by_topic(topic_str)
            except Exception:
                return label, []

        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
            cat_results = list(pool.map(_fetch_cat, GNEWS_TOPICS.items()))

        for label, news in cat_results:
            cat_counts.append(len(news))
            all_articles.extend(news)
            for art in news:
                try:
                    polarity = TextBlob(art.get("title", "")).sentiment.polarity
                    if polarity > 0.1: sentiments["Positive"] += 1
                    elif polarity < -0.1: sentiments["Negative"] += 1
                    else: sentiments["Neutral"] += 1
                except Exception:
                    sentiments["Neutral"] += 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            rss_future = pool.submit(_get_rss_trending, 20)
            try:
                trends = rss_future.result(timeout=15)
            except concurrent.futures.TimeoutError:
                logger.warning("RSS trending timed out")
                trends = []

        if not trends:
            word_trends = get_word_trends(all_articles)
            trends = [{"topic": t["topic"], "rank": t["rank"], "region": "India", "count": t["count"]} for t in word_trends]

        return jsonify({
            "success":        True,
            "total_analyzed": len(all_articles),
            "categories":     {"labels": cat_labels, "data": cat_counts},
            "trends":         trends,
            "sentiment":      {"labels": list(sentiments.keys()), "data": list(sentiments.values())},
        })

    except Exception as exc:
        logger.error(f"dashboard_data error: {exc}")
        if all_articles:
            word_trends = get_word_trends(all_articles)
            return jsonify({
                "success":        True,
                "total_analyzed": len(all_articles),
                "categories":     {"labels": cat_labels, "data": cat_counts or [0]*len(cat_labels)},
                "trends":         [{"topic": t["topic"], "rank": t["rank"], "region": "India", "count": t["count"]} for t in word_trends],
                "sentiment":      {"labels": list(sentiments.keys()), "data": list(sentiments.values())},
            })
        return jsonify({"success": False, "error": str(exc)})

@app.route("/api/health", methods=["GET"])
def health():
    try:
        from agent import get_ensemble
        classifier, reasoner, groq, credibility, aggregator = get_ensemble()
        ensemble_status = {
            "deberta":     {"available": getattr(classifier, "available", False)},
            "qwen_3b":     {"available": getattr(reasoner,   "available", False)},
            "groq":        {"available": True},
            "credibility": {"available": True},
        }
    except ImportError:
        ensemble_status = {"error": "agent.py not found or failed to load"}
        
    return jsonify({
        "status": "ok",
        "service": "local_backend",
        "ensemble": ensemble_status,
    })

if __name__ == "__main__":
    app.run(debug=False, port=5000, use_reloader=False)