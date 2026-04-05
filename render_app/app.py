"""
render_app/app.py
─────────────────
Deployed on Render. Zero ML dependencies.
Serves the frontend and proxies all verify requests
to the Kaggle ML backend via the ngrok public URL.

Environment variables on Render:
  KAGGLE_BACKEND_URL = https://xxxx-xx-xx.ngrok-free.app
    (update this every time you restart the Kaggle notebook)
"""

import os
import requests
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_THIS_DIR)

app = Flask(
    __name__,
    template_folder=os.path.join(_BASE_DIR, "templates"),
    static_folder=os.path.join(_BASE_DIR, "static"),
    static_url_path="/static",
)
CORS(app)

KAGGLE_BACKEND_URL = os.getenv("KAGGLE_BACKEND_URL", "").rstrip("/")


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
        return jsonify({"error": "ML backend not configured. Set KAGGLE_BACKEND_URL on Render."}), 503

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


# ─── Feed + Dashboard → proxy to Kaggle ───────────────────────

@app.route("/api/feed", methods=["GET"])
def feed():
    if not KAGGLE_BACKEND_URL:
        return jsonify([])
    try:
        resp = requests.get(_kaggle("/api/ml-feed"), timeout=30)
        return jsonify(resp.json())
    except Exception:
        return jsonify([])


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
