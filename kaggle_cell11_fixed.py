# ─── Cell 11: Flask ML server ────────────────────────────────
import re as re_lib
from flask import Flask, request, jsonify
from flask_cors import CORS
from gnews import GNews
from textblob import TextBlob

ml_app = Flask(__name__)
CORS(ml_app)


@ml_app.route("/api/ml-health", methods=["GET"])
def ml_health():
    return jsonify({
        "status":      "ok",
        "service":     "kaggle_ml_backend",
        "qwen_vl_8b":  qwen_model.available,   # FIX: was "qwen_3b"
        "groq":        True,
    })


@ml_app.route("/reason", methods=["POST"])
def ml_reason():
    """
    Called by KaggleQwenReasoner in backend/ensemble/kaggle_reasoner.py.
    Receives pre-scraped text + evidence from local app, runs Qwen inference.
    """
    body = request.get_json(force=True, silent=True)
    if not body:
        logger.error("/reason received empty body")
        return jsonify({"error": "No JSON body"}), 400

    text      = body.get("text", "")
    evidence  = body.get("evidence", [])
    image_url = body.get("image_url")

    if not text:
        return jsonify({"verdict": "UNVERIFIED", "confidence": 0,
                        "reasoning": "No text provided", "source": "qwen_vl_8b"})

    logger.info(f"/reason | text_len={len(text)} evidence={len(evidence)} image={'yes' if image_url else 'no'}")
    result = qwen_model.reason(text, evidence, image_url=image_url)
    return jsonify(result)


@ml_app.route("/api/ml-verify", methods=["POST"])
def ml_verify():
    """Full pipeline: scrape URL → search → both models → aggregate."""
    body = request.get_json(force=True, silent=True)
    if not body or not body.get("url"):
        return jsonify({"error": "Missing url"}), 400
    url = body["url"].strip()
    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "Invalid URL"}), 400

    content = scrape_content(url)
    if "error" in content:
        return jsonify({"error": content["error"]}), 500

    query    = generate_query(content)
    evidence = search_evidence(query)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        groq_f = pool.submit(groq_model.reason, content["text"], evidence)
        qwen_f = pool.submit(qwen_model.reason, content["text"], evidence)

    groq_r = groq_f.result()
    qwen_r = qwen_f.result()
    cred_r = score_credibility(evidence)
    final  = aggregate(groq_r, qwen_r, cred_r)

    return jsonify({
        "title":              content["title"],
        "verdict":            final["verdict"],
        "confidence":         final["confidence"],
        "status":             final["status"],
        "reasoning":          final["reasoning"],
        "sources":            evidence,
        "ensemble_breakdown": final["ensemble_breakdown"],
    })


STOP_WORDS = {"the","and","for","that","with","from","this","news","live","update",
              "says","will","after","what","where","when","google","full","coverage"}


@ml_app.route("/api/ml-feed", methods=["GET"])
def ml_feed():
    try:
        gn   = GNews(language="en", country="IN", max_results=12)
        arts = gn.get_top_news()
        return jsonify([{
            "headline": a.get("title", ""),
            "source":   a.get("publisher", {}).get("title", "Unknown"),
            "link":     a.get("url", "#"),
            "image":    a.get("image") or "https://placehold.co/300x180?text=News",
            "time":     a.get("published date", ""),
        } for a in arts])
    except Exception:
        return jsonify([])


@ml_app.route("/api/ml-dashboard", methods=["GET"])
def ml_dashboard():
    categories = ["Technology","Science","Health","Business","Entertainment"]
    cat_counts, all_arts = [], []
    sentiments = {"Positive":0,"Neutral":0,"Negative":0}
    try:
        gn = GNews(language="en", country="IN", max_results=10)
        for cat in categories:
            news = gn.get_news_by_topic(cat.upper())
            cat_counts.append(len(news))
            all_arts.extend(news)
            for a in news:
                p = TextBlob(a.get("title","")).sentiment.polarity
                sentiments["Positive" if p>0.1 else "Negative" if p<-0.1 else "Neutral"] += 1
        words = re_lib.findall(r"\w+", " ".join(a.get("title","") for a in all_arts).lower())
        words = [w for w in words if w not in STOP_WORDS and len(w) > 4]
        trends = Counter(words).most_common(10)
        return jsonify({
            "success": True, "total_analyzed": len(all_arts),
            "categories": {"labels": categories, "data": cat_counts},
            "trends":     {"labels": [t[0] for t in trends], "data": [t[1] for t in trends]},
            "sentiment":  {"labels": list(sentiments.keys()), "data": list(sentiments.values())},
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})