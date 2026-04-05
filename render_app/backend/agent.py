"""
backend/agent.py
SachBol AI — Hybrid Ensemble Agentic Pipeline

4 components, 3 run in parallel:
  ┌── DeBERTa classifier    (25%) ──┐
  ├── Qwen3-VL-8B reasoner  (25%) ──┼── aggregator → final verdict
  ├── Groq 70B reasoner     (35%) ──┤
  └── Credibility scorer    (15%) ──┘

Qwen3-VL-8B can now run on:
  1. Kaggle T4 backend via ngrok  (USE_QWEN_KAGGLE=true, preferred)
  2. Local GPU/CPU                (USE_QWEN_LOCAL=true)
  3. Inactive — weight redistrib. (both false)

New in this version:
  - Extracts og:image from scraped articles and passes to Qwen VL reasoner
  - image_url flows through run_agent → qwen.reason for multimodal verification
"""

import sys
import os
import json
import logging
import concurrent.futures
import requests
import cloudscraper
from bs4 import BeautifulSoup
from groq import Groq
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    GROQ_API_KEY, SERPER_API_KEY, MODEL_NAME,
    USE_BGE_KAGGLE, BGE_KAGGLE_URL,
    USE_BGE_RERANKER, BGE_MODEL_ID,
    USE_DEBERTA, DEBERTA_MODEL_HF_ID, DEBERTA_TEMPERATURE_PATH,
    USE_QWEN_KAGGLE, QWEN_KAGGLE_URL,
    USE_QWEN_LOCAL,  QWEN_MODEL_HF_ID,
)
from ensemble import (
    LocalClassifier, BGERankerClassifier, KaggleBGEReasoner,
    QwenVLReasoner, KaggleQwenReasoner,
    GroqReasoner, CredibilityScorer, EnsembleAggregator,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(message)s",
                    datefmt="%m/%d/%Y %I:%M:%S %p")
logger = logging.getLogger(__name__)

# ─── Singletons ───────────────────────────────────────────────
_ready        = False
_deberta      = None
_qwen         = None
_groq         = None
_credibility  = None
_aggregator   = None


def init_ensemble():
    global _ready, _deberta, _qwen, _groq, _credibility, _aggregator
    if _ready:
        return

    logger.info("Initialising ensemble components...")

    # 4th ensemble component: BGE Kaggle > BGE local > DeBERTa > Inactive
    if USE_BGE_KAGGLE and BGE_KAGGLE_URL:
        logger.info(f"4th component → BGE Reranker on Kaggle T4: {BGE_KAGGLE_URL}")
        _deberta = KaggleBGEReasoner(base_url=BGE_KAGGLE_URL)
    elif USE_BGE_RERANKER:
        logger.info(f"4th component → BGE Reranker (local): {BGE_MODEL_ID}")
        _deberta = BGERankerClassifier(model_id=BGE_MODEL_ID)
    elif USE_DEBERTA:
        logger.info(f"4th component → DeBERTa: {DEBERTA_MODEL_HF_ID}")
        _deberta = LocalClassifier(
            model_id         = DEBERTA_MODEL_HF_ID,
            temperature_path = DEBERTA_TEMPERATURE_PATH,
        )
    else:
        logger.info("4th component → inactive (set USE_BGE_RERANKER=true in .env)")
        _deberta = _make_inactive_classifier()

    # Qwen3-VL-8B — Kaggle > Local > Inactive
    if USE_QWEN_KAGGLE and QWEN_KAGGLE_URL:
        logger.info(f"Qwen VL 8B → Kaggle backend at {QWEN_KAGGLE_URL}")
        _qwen = KaggleQwenReasoner(base_url=QWEN_KAGGLE_URL)
    elif USE_QWEN_LOCAL:
        logger.info("Qwen VL 8B → loading locally (requires ~7 GB VRAM)")
        _qwen = QwenVLReasoner(model_id=QWEN_MODEL_HF_ID)   # was QwenReasoner
    else:
        _qwen = _make_inactive_qwen()

    _groq        = GroqReasoner(api_key=GROQ_API_KEY, model=MODEL_NAME)
    _credibility = CredibilityScorer()
    _aggregator  = EnsembleAggregator()
    _ready       = True
    logger.info("Ensemble ready")


def get_ensemble():
    init_ensemble()
    return _deberta, _qwen, _groq, _credibility, _aggregator


def _make_inactive_classifier():
    class _Inactive:
        available = False
        def classify(self, text, evidence=None):
            return {"verdict": "UNVERIFIED", "confidence": 0,
                    "source": "deberta_classifier"}
    return _Inactive()


def _make_inactive_qwen():
    class _Inactive:
        available = False
        def reason(self, *args, **kwargs): return {"verdict": "UNVERIFIED", "confidence": 0,
                                                    "reasoning": "Qwen VL disabled.",
                                                    "source": "qwen_vl_8b", "used_image": False}
    return _Inactive()


# ─── Stage 1: Scraping ────────────────────────────────────────

def resolve_google_news_url(url: str) -> str:
    """Follow Google News redirects to reach the real article URL."""
    if "news.google.com" not in url:
        return url
    try:
        r = requests.get(url, allow_redirects=True, timeout=8,
                         headers={"User-Agent": "Mozilla/5.0"})
        return r.url
    except Exception:
        return url


def scrape_content(url: str) -> dict:
    # Try jina.ai reader first — bypasses most paywalls and bot blocks
    try:
        r = requests.get(
            f"https://r.jina.ai/{url}",
            timeout=15,
            headers={"Accept": "text/plain"},
        )
        if r.status_code == 200 and len(r.text) > 300:
            lines = r.text.strip().split("\n")
            title = lines[0].replace("#", "").strip() if lines else "Article"
            text  = " ".join(lines[1:])[:4000].strip()
            og_image = None
            # jina doesn't give og:image — scrape it separately below
            try:
                head_resp = requests.get(url, timeout=6,
                    headers={"User-Agent": "Mozilla/5.0"})
                soup2 = BeautifulSoup(head_resp.text, "html.parser")
                og_image = (soup2.find("meta", property="og:image") or {}).get("content", "").strip() or None
            except Exception:
                pass
            usable_image = og_image if _is_usable_image(og_image) else None
            return {
                "title":     title,
                "text":      text or title,
                "platform":  _detect_platform(url),
                "image_url": usable_image,
                "media":     [usable_image] if usable_image else [],
            }
    except Exception as e:
        logger.debug(f"Jina failed ({e}) — falling back to cloudscraper")

    # Fallback: cloudscraper (existing code unchanged below)
    try:
        scraper = cloudscraper.create_scraper()
        resp    = scraper.get(url, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        title = (
            (soup.find("meta", property="og:title") or {}).get("content")
            or (soup.find("title") or {}).get_text()
            or ""
        ).strip()

        og_image = (soup.find("meta", property="og:image") or {}).get("content", "").strip()
        if not og_image:
            for img in soup.find_all("img", src=True):
                src = img["src"]
                if src.startswith("http") and not src.endswith((".ico", ".svg")):
                    og_image = src
                    break

        for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
            tag.decompose()
        text = " ".join(soup.get_text(" ", strip=True).split())[:4000]

        usable_image2 = og_image if _is_usable_image(og_image) else None
        return {
            "title":     title,
            "text":      text or title,
            "platform":  _detect_platform(url),
            "image_url": usable_image2,
            "media":     [usable_image2] if usable_image2 else [],
        }
    except Exception as exc:
        logger.error(f"Scrape error ({url}): {exc}")
        return {"error": str(exc)}


_BAD_IMAGE_PATTERNS = (
    "news.google.com",
    "google.com/s2",
    "placehold.co",
    "placeholder",
    "1x1",
    "pixel",
    "/logo",
    "logo.",
    "icon.",
    "favicon",
)

def _is_usable_image(url: str | None) -> bool:
    """Return True only if the URL looks like a real article image."""
    if not url:
        return False
    url_lower = url.lower()
    for bad in _BAD_IMAGE_PATTERNS:
        if bad in url_lower:
            return False
    return True


def _detect_platform(url: str) -> str | None:
    domain = urlparse(url).netloc.lower()
    for kw, name in [
        ("twitter.com", "twitter"), ("x.com", "twitter"),
        ("instagram.com", "instagram"), ("facebook.com", "facebook"),
        ("youtube.com", "youtube"), ("youtu.be", "youtube"),
        ("whatsapp", "whatsapp"),
    ]:
        if kw in domain:
            return name
    return None


# ─── Stage 2: Query generation ────────────────────────────────

def generate_search_query(content: dict, hint_title: str = "") -> str:
    title = hint_title or content.get("title", "")
    text  = content.get("text", "")
    base  = title or text[:120]
    # Strip common filler words for a tighter query
    words = [w for w in base.split() if len(w) > 3][:10]
    return " ".join(words)


# ─── Stage 3: Evidence retrieval ──────────────────────────────

def search_evidence(query: str) -> list:
    logger.info(f"Searching: {query}")
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 8, "gl": "in"},
            timeout=10,
        )
        return [
            {"title": i.get("title",""), "link": i.get("link",""), "snippet": i.get("snippet","")}
            for i in resp.json().get("organic", [])
        ]
    except Exception as exc:
        logger.error(f"Search error: {exc}")
        return []


# ─── Main pipeline ────────────────────────────────────────────

def run_agent(url: str, hint_title: str = "") -> dict:
    """
    Full 4-component hybrid ensemble pipeline.

    hint_title: headline text passed from the feed card's Verify button.
    image_url:  extracted from og:image and passed to Qwen VL reasoner for
                multimodal fact-checking of the article's primary image.
    """
    # Stage 0: resolve Google News redirects
    resolved_url = resolve_google_news_url(url)

    # Stage 1: scrape
    content = scrape_content(resolved_url)

    if "error" in content:
        if hint_title:
            logger.info(f"Scrape failed — using hint_title: {hint_title[:80]}")
            content = {"title": hint_title, "text": hint_title, "image_url": None}
        else:
            return {"error": content["error"]}

    article_text  = content["text"]

    # Use hint_title if the scraped title looks like a platform/hosting name
    def _is_bad_title(t: str) -> bool:
        tl = t.lower().strip()
        if not tl or len(tl) < 5:
            return True
        bad_substrings = ("google news", "gnews", "news.google", "unknown", "article")
        return any(b in tl for b in bad_substrings)

    scraped_title = content.get("title", "").strip()
    article_title = (
        hint_title or scraped_title or "Unknown"
        if _is_bad_title(scraped_title)
        else scraped_title or hint_title or "Unknown"
    )

    # Only pass image_url to Qwen if it's a real article image (not a logo/placeholder)
    image_url = content.get("image_url")   # already filtered by _is_usable_image in scrape_content

    # Stage 2+3
    query    = generate_search_query(content, hint_title=hint_title)
    evidence = search_evidence(query)

    if (len(article_text) < 150 or article_text == hint_title) and evidence:
        enriched = " ".join(
            f"{e.get('title', '')}. {e.get('snippet', '')}" for e in evidence[:4]
        )
        article_text = f"{article_title}\n\n{enriched}"
        logger.info(f"Short article text ({len(content['text'])} chars) — enriched with evidence snippets")

    # Stage 4: Ensemble — all three models run in parallel
    deberta, qwen, groq, credibility_scorer, aggregator = get_ensemble()

    logger.info("Running 4-component ensemble in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        groq_future    = pool.submit(groq.reason,      article_text, evidence)
        # BGE reranker needs evidence to score relevance — pass it through
        deberta_future = pool.submit(deberta.classify,  article_text, evidence)
        # Pass image_url to Qwen VL — it will use it if the model supports vision
        qwen_future    = pool.submit(qwen.reason,       article_text, evidence, image_url)

    try:
        groq_result = groq_future.result(timeout=45)
    except concurrent.futures.TimeoutError:
        logger.error("Groq 70B timed out")
        groq_result = {"verdict": "UNVERIFIED", "confidence": 0,
                       "reasoning": "Groq API timed out.", "key_sources": [], "source": "groq"}
    except Exception as exc:
        logger.error(f"Groq 70B error: {exc}")
        groq_result = {"verdict": "UNVERIFIED", "confidence": 0,
                       "reasoning": str(exc), "key_sources": [], "source": "groq"}

    try:
        deberta_result = deberta_future.result(timeout=15)
    except concurrent.futures.TimeoutError:
        logger.error("DeBERTa timed out")
        deberta_result = {"verdict": "UNVERIFIED", "confidence": 0, "source": "deberta_classifier"}
    except Exception as exc:
        logger.error(f"DeBERTa error: {exc}")
        deberta_result = {"verdict": "UNVERIFIED", "confidence": 0, "source": "deberta_classifier"}

    try:
        qwen_result = qwen_future.result(timeout=50)   # +5s for VL image processing
    except concurrent.futures.TimeoutError:
        logger.error("Qwen VL 8B timed out")
        qwen_result = {"verdict": "UNVERIFIED", "confidence": 0,
                       "reasoning": "Qwen VL timed out.", "source": "qwen_vl_8b", "used_image": False}
    except Exception as exc:
        logger.error(f"Qwen VL 8B error: {exc}")
        qwen_result = {"verdict": "UNVERIFIED", "confidence": 0,
                       "reasoning": str(exc), "source": "qwen_vl_8b", "used_image": False}

    credibility_result = credibility_scorer.score(evidence)

    # Stage 5: Aggregate
    final = aggregator.aggregate(
        groq_result, deberta_result, qwen_result, credibility_result
    )

    return {
        "title":              article_title,
        "verdict":            final["verdict"],
        "confidence":         final["confidence"],
        "status":             final["status"],
        "reasoning":          final["reasoning"],
        "groq_reasoning":     groq_result.get("reasoning", ""),
        "qwen_reasoning":     qwen_result.get("reasoning", ""),
        "sources":            evidence,
        "ensemble_breakdown": final["ensemble_breakdown"],
        "platform":           content.get("platform", None),
        "media":              content.get("media", []),
        "image_analysed":     qwen_result.get("used_image", False),
    }
