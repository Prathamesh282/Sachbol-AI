# SachBol AI 🛡️

**India's fact-checking engine** — a 4-model ensemble that scrapes any news URL, retrieves live Google evidence, and returns a calibrated verdict in seconds.

```
Ensemble composition:
  Groq LLaMA-3.3-70B  →  40%   (accuracy + key sources)
  DeBERTa-v3-Large    →  30%   (fast calibrated classifier)
  Qwen-2.5-3B         →  15%   (evidence-grounded reasoning)
  Credibility Scorer  →  15%   (source reputation heuristics)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  LOCAL DEVELOPMENT                                          │
│  python -m backend.app  →  runs full ML stack on port 5000  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  PRODUCTION (distributed)                                   │
│                                                             │
│  User → Render (render_app/app.py)                          │
│              ↓  proxies ML calls via ngrok                  │
│         Kaggle (kaggle_server.py)                           │
│              ↓  loads models from HuggingFace               │
│         Qwen 3B adapter + Groq API                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
sachbol/
├── backend/
│   ├── app.py              ← Flask server (full ML, local dev)
│   ├── config.py           ← Env var loading
│   ├── agent.py            ← 4-stage pipeline orchestrator
│   └── ensemble/
│       ├── __init__.py     ← Package exports
│       ├── aggregator.py   ← Weighted voting logic
│       ├── classifier.py   ← DeBERTa-v3 classifier
│       ├── credibility.py  ← Source trust heuristics
│       ├── groq_reasoner.py← Groq LLaMA-70B reasoning
│       └── reasoner.py     ← Qwen 3B local reasoning
├── render_app/
│   ├── app.py              ← Lightweight gateway (Render deploy)
│   └── requirements.txt    ← Render deps (no ML)
├── templates/
│   ├── index.html          ← Main verifier UI
│   └── dashboard.html      ← Analytics dashboard
├── static/
│   ├── style.css
│   ├── script.js
│   └── dashboard.js
├── training/
│   ├── dataset.py          ← LIAR dataset prep
│   └── train.py            ← Fine-tuning script
├── kaggle_server.py        ← Full ML backend (run in Kaggle)
├── requirements.txt        ← Full deps (local dev)
├── render.yaml             ← Render build config
├── Dockerfile              ← Docker (local dev)
├── .env.example            ← Env var template
└── .gitignore
```

---

## Running Locally (Quickstart)

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com) (free)
- A [Serper API key](https://serper.dev) (free, 2500 searches/month)

### Step 1 — Clone & setup

```bash
git clone https://github.com/yourname/sachbol.git
cd sachbol

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### Step 2 — Install dependencies

**CPU-only (works everywhere, slower inference):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**GPU (CUDA, faster):**
```bash
pip install torch                # pulls CUDA build automatically
pip install -r requirements.txt
```

> **Tip:** If you only want to run with Groq (no local DeBERTa/Qwen models),
> set `USE_DEBERTA=false` and `USE_QWEN_LOCAL=false` in your `.env`.
> The app still works — Groq + Credibility carry 55% of the ensemble weight.

### Step 3 — Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in at minimum:
```
GROQ_API_KEY=gsk_...
SERPER_API_KEY=...
```

### Step 4 — Run

```bash
python -m backend.app
```

Open http://localhost:5000 — that's it.

---

## Running with Docker

```bash
docker build -t sachbol .
docker run --env-file .env -p 5000:5000 sachbol
```

Open http://localhost:5000

---

## Disabling Heavy Local Models (Groq-only mode)

If you don't have GPU / don't want to download the 3B model, add to `.env`:

```
USE_DEBERTA=false
USE_QWEN_LOCAL=false
```

The app runs entirely on Groq 70B + Credibility scorer.
Verdicts are still high-quality — just without the local model contributions.

---

## Production Deployment (Render + Kaggle)

This distributed setup runs heavy ML on Kaggle's free T4 GPU
and routes requests through Render's lightweight gateway.

### Step 1 — Add Kaggle Secrets

In your Kaggle notebook → Add-ons → Secrets:
| Key | Value |
|---|---|
| `GROQ_API_KEY` | your Groq key |
| `SERPER_API_KEY` | your Serper key |
| `HF_TOKEN` | your HuggingFace token |
| `NGROK_AUTH_TOKEN` | from https://dashboard.ngrok.com |

### Step 2 — Push to GitHub

```bash
git init
git add .
git commit -m "initial"
git remote add origin https://github.com/yourname/sachbol
git push -u origin main
```

### Step 3 — Deploy to Render

1. Go to [render.com](https://render.com) → **New Web Service**
2. Connect your GitHub repo
3. Build command: `pip install -r render_app/requirements.txt`
4. Start command: `gunicorn -w 2 -b 0.0.0.0:$PORT render_app.app:app`
5. Add env var: `KAGGLE_BACKEND_URL` = *(leave blank for now)*

### Step 4 — Start Kaggle ML backend

1. Open `kaggle_server.py` in a new Kaggle notebook
2. Enable **GPU (T4)** and **Internet** in notebook settings
3. Uncomment and run the `!pip install` cell
4. Run all cells — copy the ngrok URL printed at the end

### Step 5 — Connect Render → Kaggle

1. Render dashboard → your service → **Environment**
2. Set `KAGGLE_BACKEND_URL` = `https://xxxx-xx-xx.ngrok-free.app`
3. Render redeploys automatically (~30 seconds)

> ⚠️ **Every time you restart the Kaggle notebook**, the ngrok URL changes.
> Repeat steps 4–5. Upgrade to a paid ngrok plan for a fixed domain.

---

## API Reference

### `POST /api/verify`

Fact-checks a news article URL.

**Request:**
```json
{ "url": "https://example.com/news-article" }
```

**Response:**
```json
{
  "title":     "Article headline",
  "verdict":   "VERIFIED | MOSTLY_TRUE | MOSTLY_FALSE | FALSE | UNVERIFIED",
  "confidence": 84,
  "status":    "safe | caution | danger | unknown",
  "reasoning": "Two independent sources confirm...",
  "sources":   [{ "title": "...", "link": "...", "snippet": "..." }],
  "ensemble_breakdown": {
    "groq":   { "verdict": "VERIFIED", "confidence": 88, "weight": 0.4 },
    "deberta":    { "verdict": "VERIFIED", "confidence": 79, "weight": 0.3 },
    "qwen_3b":    { "verdict": "MOSTLY_TRUE", "confidence": 72, "weight": 0.15 },
    "credibility":{ "score": 82, "trusted_sources": 4, "debunked": false }
  }
}
```

### `GET /api/feed`
Returns top 12 Indian news headlines.

### `GET /api/dashboard-data`
Returns sentiment, topic distribution, and trending keywords.

### `GET /api/health`
Returns ensemble component availability status.

---

## HuggingFace Models

| Model | HF ID | Role |
|---|---|---|
| DeBERTa-v3 | `prathameshbandal/sachbol-deberta-classifier` | Fast classifier (30%) |
| Qwen adapter | `prathameshbandal/sachbol-qwen-reasoner` | Reasoning (15%) |

---

## Ensemble Weights

| Component | Weight | Notes |
|---|---|---|
| Groq LLaMA-3.3-70B | 40% | Accuracy + key sources |
| DeBERTa-v3-Large | 30% | Fast, temperature-calibrated |
| Qwen-2.5-3B | 15% | Evidence-grounded reasoning |
| Credibility Scorer | 15% | Domain reputation heuristics |

Weights are renormalized automatically if local models are unavailable.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'ensemble'`**
→ Run from project root: `python -m backend.app` (not `cd backend && python app.py`)

**`GROQ_API_KEY not set`**
→ Make sure `.env` exists in project root and contains `GROQ_API_KEY=...`

**`TemplateNotFoundError: index.html`**
→ Make sure `templates/` folder is at project root (not inside `backend/`)

**Local models not loading**
→ Set `USE_DEBERTA=false` and `USE_QWEN_LOCAL=false` in `.env` to run Groq-only

**Kaggle ngrok URL expired**
→ Restart Kaggle notebook, copy new URL, update `KAGGLE_BACKEND_URL` on Render

---

## License

MIT — build on it, improve it, share it.
