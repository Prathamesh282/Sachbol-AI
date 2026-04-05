# SachBol AI 🛡️

**India's fact-checking engine** — a 3-model ensemble that scrapes any news URL, retrieves live Google evidence, and returns a calibrated verdict in seconds.

```
Ensemble composition (Kaggle/Production):
  Groq LLaMA-4 Scout 17B  →  50%   (accuracy + key sources)
  Qwen3-VL-8B             →  35%   (multimodal evidence reasoning)
  Credibility Scorer      →  15%   (source reputation heuristics)

Optional local-only component:
  BGE Reranker v2-m3      →  replaces Credibility when USE_BGE_RERANKER=true
  DeBERTa-v3-Large        →  legacy classifier (fine-tuned checkpoint required)
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
│              ↓  loads Qwen3-VL-8B from HuggingFace          │
│         Qwen3-VL-8B adapter + Groq API                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
sachbol/
├── backend/
│   ├── app.py              ← Flask server (full ML, local dev)
│   ├── config.py           ← Env var loading
│   └── ensemble/
│       ├── __init__.py     ← Package exports
│       ├── aggregator.py   ← Weighted voting logic
│       ├── credibility.py  ← Source trust heuristics
│       ├── groq_reasoner.py← Groq LLaMA-4 Scout reasoning
│       ├── kaggle_bge.py   ← BGE Reranker via Kaggle backend
│       └── reasoner.py     ← Qwen3-VL-8B local reasoning
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
git clone https://github.com/Prathamesh282/sachbol.git
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

> **Tip:** If you only want to run with Groq (no local Qwen/BGE models),
> set `USE_QWEN_LOCAL=false` and `USE_BGE_RERANKER=false` in your `.env`.
> The app still works — Groq + Credibility carry 65%+ of the ensemble weight.

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

If you don't have a GPU / don't want to download large models, add to `.env`:

```
USE_QWEN_LOCAL=false
USE_BGE_RERANKER=false
USE_DEBERTA=false
```

The app runs entirely on Groq LLaMA-4 Scout + Credibility scorer.
Verdicts are still high quality — just without the local model contributions.

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
git remote add origin https://github.com/Prathamesh282/sachbol
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
    "groq":       { "verdict": "VERIFIED", "confidence": 88, "weight": 0.50 },
    "qwen_vl_8b": { "verdict": "VERIFIED", "confidence": 79, "weight": 0.35, "available": true },
    "credibility":{ "score": 82, "trusted_sources": 4, "debunked": false, "weight": 0.15 }
  }
}
```

### `GET /api/feed`
Returns top 12 Indian news headlines.

### `GET /api/dashboard-data`
Returns sentiment, topic distribution, and trending keywords.

### `GET /api/health`
Returns ensemble component availability status.

### `POST /rerank` *(Kaggle backend only)*
BGE Reranker endpoint — scores evidence passages for relevance to a claim.

---

## HuggingFace Models

| Model | HF ID | Role |
|---|---|---|
| Qwen3-VL-8B adapter | `prathameshbandal/sachbol-qwen3vl-final` | Multimodal evidence reasoning (35%) |
| DeBERTa-v3 (legacy) | `prathameshbandal/sachbol-deberta-classifier` | Optional local classifier |

---

## Ensemble Weights

### Kaggle / Production

| Component | Weight | Notes |
|---|---|---|
| Groq LLaMA-4 Scout 17B | 50% | Accuracy + key sources |
| Qwen3-VL-8B | 35% | Multimodal evidence-grounded reasoning |
| Credibility Scorer | 15% | Domain reputation heuristics |

Weights are renormalized automatically if a component is unavailable.

### Local Development (optional overrides)

| Component | Env flag | Notes |
|---|---|---|
| BGE Reranker v2-m3 | `USE_BGE_RERANKER=true` | Replaces Credibility scorer locally |
| BGE via Kaggle | `USE_BGE_KAGGLE=true` + `BGE_KAGGLE_URL=...` | Same ngrok URL as Qwen |
| DeBERTa-v3-Large | `USE_DEBERTA=true` | Requires fine-tuned checkpoint |
| Qwen3-VL-8B local | `USE_QWEN_LOCAL=true` + `QWEN_KAGGLE_URL=...` | ~7 GB VRAM (bfloat16) or ~4 GB (4-bit) |

---

## Environment Variables

```env
# Required
GROQ_API_KEY=gsk_...
SERPER_API_KEY=...

# Groq model (default: Llama 4 Scout)
GROQ_MODEL_NAME=meta-llama/llama-4-scout-17b-16e-instruct

# Qwen — choose ONE
USE_QWEN_KAGGLE=true
QWEN_KAGGLE_URL=https://xxxx.ngrok.io   # your Kaggle ngrok URL
# OR
USE_QWEN_LOCAL=false
QWEN_MODEL_HF_ID=prathameshbandal/sachbol-qwen3vl-final

# 4th component — choose ONE (BGE Kaggle > BGE local > DeBERTa)
USE_BGE_KAGGLE=false
BGE_KAGGLE_URL=https://xxxx.ngrok.io    # can reuse QWEN_KAGGLE_URL
USE_BGE_RERANKER=false
USE_DEBERTA=false

# Production (Render gateway)
KAGGLE_BACKEND_URL=https://xxxx-xx-xx.ngrok-free.app
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'ensemble'`**
→ Run from project root: `python -m backend.app` (not `cd backend && python app.py`)

**`GROQ_API_KEY not set`**
→ Make sure `.env` exists in project root and contains `GROQ_API_KEY=...`

**`TemplateNotFoundError: index.html`**
→ Make sure `templates/` folder is at project root (not inside `backend/`)

**Local models not loading**
→ Set `USE_QWEN_LOCAL=false`, `USE_BGE_RERANKER=false`, `USE_DEBERTA=false` in `.env` to run Groq-only

**Kaggle ngrok URL expired**
→ Restart Kaggle notebook, copy new URL, update `KAGGLE_BACKEND_URL` on Render and `QWEN_KAGGLE_URL` in `.env`

**Qwen3-VL-8B OOM on T4**
→ The model is loaded in 4-bit NF4 quantization by default (~5 GB VRAM). If you still hit OOM, make sure no other notebooks are running on the same GPU session.

---

## License

MIT — build on it, improve it, share it.
