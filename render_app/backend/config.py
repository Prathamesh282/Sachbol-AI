"""
backend/config.py

Add these to .env:
  GROQ_API_KEY=gsk_...
  SERPER_API_KEY=...

  # 4th ensemble component — choose ONE (BGE Kaggle takes priority over all):
  # Option A: BGE Reranker on Kaggle T4 (recommended — no local disk/VRAM needed)
  USE_BGE_KAGGLE=true
  BGE_KAGGLE_URL=https://xxxx.ngrok.io   # same URL as QWEN_KAGGLE_URL is fine

  # Option B: BGE Reranker locally (requires ~1.1 GB VRAM/disk)
  USE_BGE_RERANKER=false
  BGE_MODEL_ID=BAAI/bge-reranker-v2-m3

  # Option C: DeBERTa classifier (requires fine-tuned checkpoint)
  USE_DEBERTA=false
  DEBERTA_MODEL_HF_ID=prathameshbandal/sachbol-deberta-classifier

  # Qwen3-VL-8B — choose ONE mode:
  USE_QWEN_KAGGLE=true               # run on Kaggle T4 via ngrok (recommended)
  QWEN_KAGGLE_URL=https://xxxx.ngrok.io

  # OR run locally (requires ~7 GB VRAM for bfloat16, ~4 GB for 4-bit):
  USE_QWEN_LOCAL=false
  QWEN_MODEL_HF_ID=prathameshbandal/sachbol-qwen3vl-final
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

if not GROQ_API_KEY:
    logger.warning(
        "GROQ_API_KEY not set — /api/verify will fail. "
        "Add it to sachbol/.env  (see .env.example)"
    )

GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct")
MODEL_NAME      = GROQ_MODEL_NAME

# ── 4th ensemble component ─────────────────────────────────────
# Priority: BGE Kaggle > BGE local > DeBERTa > Inactive

# Option A: BGE on Kaggle T4 (no local model download needed)
USE_BGE_KAGGLE = os.getenv("USE_BGE_KAGGLE", "false").lower() == "true"
BGE_KAGGLE_URL = os.getenv("BGE_KAGGLE_URL", "")   # can reuse QWEN_KAGGLE_URL

# Option B: BGE Reranker locally
USE_BGE_RERANKER = os.getenv("USE_BGE_RERANKER", "true").lower() == "true"
BGE_MODEL_ID     = os.getenv("BGE_MODEL_ID", "BAAI/bge-reranker-v2-m3")

# Option C: DeBERTa (legacy — kept for deployments with the fine-tuned checkpoint)
USE_DEBERTA              = os.getenv("USE_DEBERTA", "false").lower() == "true"
DEBERTA_MODEL_HF_ID      = os.getenv("DEBERTA_MODEL_HF_ID", "prathameshbandal/sachbol-deberta-classifier")
DEBERTA_TEMPERATURE_PATH = os.getenv("DEBERTA_TEMPERATURE_PATH", "./sachbol-deberta-final/temperature.json")

# ── Qwen3-VL-8B — Kaggle backend (HTTP) takes priority over local ──
USE_QWEN_KAGGLE  = os.getenv("USE_QWEN_KAGGLE",  "false").lower() == "true"
QWEN_KAGGLE_URL  = os.getenv("QWEN_KAGGLE_URL",  "")   # your ngrok URL

USE_QWEN_LOCAL   = os.getenv("USE_QWEN_LOCAL",   "false").lower() == "true"
QWEN_MODEL_HF_ID = os.getenv(
    "QWEN_MODEL_HF_ID",
    "prathameshbandal/sachbol-qwen3vl-final"
)
