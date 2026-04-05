# ============================================================
# kaggle_server.py  — SachBol AI Kaggle ML backend
# Run this as a Kaggle notebook (T4 GPU + Internet enabled).
#
# Ensemble on Kaggle:
#   Groq LLaMA Scout 17B  35%  — accuracy + key sources
#   Qwen3-VL-8B           25%  — multimodal evidence reasoning
#   Credibility           15%  — source reputation heuristics
#   (DeBERTa runs locally — not loaded here)
# ============================================================

# ─── Cell 1: Install ────────────────────────────────────────
# !pip install -q flask flask-cors pyngrok groq requests \
#     cloudscraper beautifulsoup4 gnews textblob feedparser \
#     "transformers>=4.57.0" peft accelerate bitsandbytes sentencepiece \
#     python-dotenv qwen-vl-utils pillow

# ─── Cell 2: Secrets + Config ───────────────────────────────
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()

GROQ_API_KEY     = secrets.get_secret("GROQ_API_KEY")
SERPER_API_KEY   = secrets.get_secret("SERPER_API_KEY")
NGROK_AUTH_TOKEN = secrets.get_secret("NGROK_AUTH_TOKEN")
HF_TOKEN         = secrets.get_secret("HF_TOKEN")

QWEN_BASE_ID       = "Qwen/Qwen3-VL-8B-Instruct"
QWEN_ADAPTER_HF_ID = "prathameshbandal/sachbol-qwen3vl-final"  # set None if not trained yet
GROQ_MODEL_NAME    = "meta-llama/llama-4-scout-17b-16e-instruct"

# ─── Cell 3: Imports ────────────────────────────────────────
import os, json, re, logging, threading, requests as req_lib
import concurrent.futures
from collections import Counter
from urllib.parse import urlparse

import torch
import cloudscraper
from bs4 import BeautifulSoup
from groq import Groq
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)
from peft import PeftModel
from huggingface_hub import login, repo_exists

login(token=HF_TOKEN)
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("sachbol")

# ─── Cell 4: T4-safe Qwen3-VL-8B loader ────────────────────
# T4 = 16 GB VRAM, float16 only (bfloat16 not supported on T4)
# 4-bit NF4 brings model to ~5 GB — leaves ~11 GB for activations

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.float16,   # float16 for T4
    bnb_4bit_use_double_quant = True,
)

QWEN_SYSTEM = """You are SachBol, an expert fact-checking AI for Indian and world news.
STRICT RULES:
1. Use ONLY the evidence provided to reach your verdict.
2. Do NOT use external knowledge not present in the evidence.
3. If evidence is insufficient → verdict must be UNVERIFIED.
4. Respond ONLY with valid JSON.
Verdicts: VERIFIED | MOSTLY_TRUE | MOSTLY_FALSE | FALSE | UNVERIFIED"""


class QwenAdapter:
    def __init__(self):
        self.available = False
        self.model     = None
        self.processor = None
        self._load()

    def _load(self):
        try:
            # Check VRAM first
            import subprocess
            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            ).stdout.strip()
            logger.info(f"GPU: {gpu}")

            logger.info(f"Loading base: {QWEN_BASE_ID}")
            base = AutoModelForImageTextToText.from_pretrained(
                QWEN_BASE_ID,
                quantization_config = bnb_config,
                device_map          = "auto",
                torch_dtype         = torch.float16,
                trust_remote_code   = True,
                low_cpu_mem_usage   = True,
            )

            # Load adapter only if it exists on HuggingFace
            # If training hasn't completed yet, skip adapter and use base model directly
            adapter_exists = (
                QWEN_ADAPTER_HF_ID is not None and
                repo_exists(QWEN_ADAPTER_HF_ID, token=HF_TOKEN)
            )

            if adapter_exists:
                logger.info(f"Loading adapter: {QWEN_ADAPTER_HF_ID}")
                self.model = PeftModel.from_pretrained(
                    base, QWEN_ADAPTER_HF_ID,
                    torch_dtype = torch.float16,
                )
                logger.info("Adapter loaded — using fine-tuned SachBol reasoning")
            else:
                logger.warning(
                    f"Adapter '{QWEN_ADAPTER_HF_ID}' not found on HuggingFace — "
                    "using base Qwen3-VL-8B. Run train_vl.py and push adapter to enable fine-tuning."
                )
                self.model = base   # base model still useful for fact-checking

            self.model.eval()

            processor_id = QWEN_ADAPTER_HF_ID if adapter_exists else QWEN_BASE_ID
            self.processor = AutoProcessor.from_pretrained(
                processor_id,
                trust_remote_code = True,
                min_pixels        = 256 * 28 * 28,
                max_pixels        = 512 * 28 * 28,   # conservative for T4
            )
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

            self.available = True
            status = "fine-tuned" if adapter_exists else "base model (adapter not trained yet)"
            logger.info(f"Qwen3-VL-8B ready on T4 — {status}")

        except torch.cuda.OutOfMemoryError:
            logger.error("OOM — try QWEN_BASE_ID = 'Qwen/Qwen3-VL-4B-Instruct' to reduce VRAM")
        except Exception as e:
            logger.warning(f"Qwen unavailable: {e}")

    def reason(self, article_text: str, evidence: list,
               image_url: str | None = None) -> dict:
        if not self.available:
            return {"verdict": "UNVERIFIED", "confidence": 0,
                    "reasoning": "Model not loaded.", "source": "qwen_vl_8b", "used_image": False}
        try:
            ev_lines = []
            for item in evidence[:5]:
                t = item.get("title", "").strip()
                s = item.get("snippet", "").strip()
                if t:
                    ev_lines.append(f"- {t}: {s[:120]}" if s else f"- {t}")

            prompt = (
                f"Fact-check this claim using ONLY the evidence provided.\n\n"
                f"CLAIM: {article_text[:800]}\n\n"
                f"EVIDENCE:\n" + "\n".join(ev_lines) + "\n\n"
                if ev_lines else
                f"Fact-check this claim. No external evidence available.\n\n"
                f"CLAIM: {article_text[:800]}\n\n"
            ) + '{"verdict": "VERIFIED|MOSTLY_TRUE|MOSTLY_FALSE|FALSE|UNVERIFIED", "confidence": <0-100>, "reasoning": "<cite evidence>"}'

            if image_url:
                user_content = [
                    {"type": "image", "image": image_url},
                    {"type": "text",  "text": prompt},
                ]
            else:
                user_content = prompt

            messages = [
                {"role": "system", "content": QWEN_SYSTEM},
                {"role": "user",   "content": user_content},
            ]

            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            if image_url:
                try:
                    from qwen_vl_utils import process_vision_info
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.processor(
                        text=[text_input], images=image_inputs, videos=video_inputs,
                        return_tensors="pt", truncation=True, max_length=1536,
                    ).to(self.model.device)
                except Exception as ve:
                    logger.warning(f"Image processing failed ({ve}) — text-only fallback")
                    image_url = None
                    inputs = self.processor.tokenizer(
                        text_input, return_tensors="pt",
                        truncation=True, max_length=1536,
                    ).to(self.model.device)
            else:
                inputs = self.processor.tokenizer(
                    text_input, return_tensors="pt",
                    truncation=True, max_length=1536,
                ).to(self.model.device)

            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens = 200,
                    do_sample      = False,
                    pad_token_id   = self.processor.tokenizer.eos_token_id,
                    use_cache      = True,
                )

            raw = self.processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            s, e = raw.find("{"), raw.rfind("}") + 1
            if s >= 0 and e > s:
                data = json.loads(raw[s:e])
                v = str(data.get("verdict", "UNVERIFIED")).upper()
                if v not in {"VERIFIED", "MOSTLY_TRUE", "MOSTLY_FALSE", "FALSE", "UNVERIFIED"}:
                    v = "UNVERIFIED"
                return {
                    "verdict":    v,
                    "confidence": max(0, min(100, int(data.get("confidence", 50)))),
                    "reasoning":  str(data.get("reasoning", ""))[:600],
                    "source":     "qwen_vl_8b",
                    "used_image": bool(image_url),
                }
        except torch.cuda.OutOfMemoryError:
            logger.error("OOM during inference — clearing cache")
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Qwen inference error: {e}")

        return {"verdict": "UNVERIFIED", "confidence": 0,
                "reasoning": "Inference failed.", "source": "qwen_vl_8b", "used_image": False}


# ─── Cell 6: Groq ───────────────────────────────────────────
GROQ_SYSTEM = """You are TruthGuard, an elite fact-checking AI.
RULES: 2+ reputable sources confirm → VERIFIED. Sources contradict → FALSE.
Fact-checkers flag hoax → FALSE. No evidence → UNVERIFIED.
Output ONLY valid JSON."""

GROQ_TMPL = """Verify this article using evidence.
ARTICLE: {article}
EVIDENCE: {evidence}
JSON: {{"verdict":"VERIFIED|MOSTLY_TRUE|MOSTLY_FALSE|FALSE|UNVERIFIED",
"confidence":<0-100>,"reasoning":"<cite sources>","key_sources":["domain1"]}}"""

class GroqReasoner:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def reason(self, article_text: str, evidence: list) -> dict:
        try:
            prompt = GROQ_TMPL.format(
                article  = article_text[:1200],
                evidence = json.dumps(evidence[:6], indent=2),
            ) if evidence else (
                f"Analyze for misinformation (no evidence available).\n"
                f"ARTICLE: {article_text[:800]}\n"
                f'JSON: {{"verdict":"...","confidence":<0-60>,"reasoning":"...","key_sources":[]}}'
            )
            resp = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": GROQ_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                model=GROQ_MODEL_NAME,
                response_format={"type": "json_object"},
                temperature=0.1, max_tokens=512,
            )
            data = json.loads(resp.choices[0].message.content)
            return {
                "verdict":     str(data.get("verdict", "UNVERIFIED")).upper(),
                "confidence":  max(0, min(100 if evidence else 60, int(data.get("confidence", 50)))),
                "reasoning":   str(data.get("reasoning", ""))[:500],
                "key_sources": data.get("key_sources", []),
                "source":      "groq",
            }
        except Exception as e:
            logger.error(f"Groq error: {e}")
            return {"verdict": "UNVERIFIED", "confidence": 0,
                    "reasoning": str(e), "key_sources": [], "source": "groq"}


# ─── Cell 7: Credibility scorer ─────────────────────────────
DOMAIN_SCORES = {
    "pib.gov.in":95,"rbi.org.in":95,"who.int":95,"reuters.com":95,"apnews.com":95,
    "bbc.com":93,"thehindu.com":87,"indianexpress.com":85,"ndtv.com":84,
    "hindustantimes.com":82,"livemint.com":82,"timesofindia.indiatimes.com":80,
    "pti.in":88,"ani.com":82,"altnews.in":82,"boomlive.in":82,"factchecker.in":82,
    "scroll.in":76,"thewire.in":74,"news18.com":72,"zeenews.india.com":62,
    "opindia.com":38,"postcard.news":22,
}
FACT_CHECK_DOMAINS = {"factcheck.org","snopes.com","boomlive.in","altnews.in",
                      "factchecker.in","vishvasnews.com","newschecker.in"}
DEBUNK_KW  = ["fake","hoax","false claim","misinformation","fabricated","debunked","no evidence"]
CONFIRM_KW = ["confirmed","government confirms","official statement","verified"]

def score_credibility(evidence: list) -> dict:
    if not evidence:
        return {"credibility_score":30,"verdict_signal":"UNVERIFIED",
                "trusted_count":0,"factcheck_found":False,"debunked":False,"confirmed":False}
    scores, debunked, confirmed, factcheck_found, trusted = [], False, False, False, 0
    for item in evidence:
        url  = item.get("link", "")
        text = (item.get("title","") + " " + item.get("snippet","")).lower()
        try:
            domain = urlparse(url).netloc.lower().removeprefix("www.")
        except Exception:
            domain = ""
        if any(k in text for k in DEBUNK_KW):  debunked  = True
        if any(k in text for k in CONFIRM_KW): confirmed = True
        if domain in FACT_CHECK_DOMAINS:        factcheck_found = True; s = 82
        else:                                   s = DOMAIN_SCORES.get(domain, 50)
        if s >= 75: trusted += 1
        scores.append(s)
    avg = int(sum(scores) / len(scores))
    if debunked:                              sig = "FALSE"
    elif factcheck_found and confirmed:       sig = "VERIFIED"
    elif trusted >= 3 or (confirmed and avg >= 75): sig = "VERIFIED"
    elif trusted >= 1 and avg >= 65:          sig = "MOSTLY_TRUE"
    elif avg < 45:                            sig = "MOSTLY_FALSE"
    else:                                     sig = "UNVERIFIED"
    return {"credibility_score":avg,"verdict_signal":sig,"trusted_count":trusted,
            "factcheck_found":factcheck_found,"debunked":debunked,"confirmed":confirmed}


# ─── Cell 8: Aggregator (Kaggle-local 3-component version) ──
# DeBERTa is not loaded on Kaggle — weights renormalized to 3 components
# FIX: all key names now consistent — "qwen_vl_8b" throughout
WEIGHTS_KAGGLE = {"groq": 0.50, "qwen_vl_8b": 0.35, "credibility": 0.15}

VERDICT_STATUS = {"VERIFIED":"safe","MOSTLY_TRUE":"caution","MOSTLY_FALSE":"caution",
                  "FALSE":"danger","UNVERIFIED":"unknown"}
VALID   = {"VERIFIED","MOSTLY_TRUE","MOSTLY_FALSE","FALSE","UNVERIFIED"}
ALIASES = {"TRUE":"VERIFIED","REAL":"VERIFIED","FAKE":"FALSE","UNKNOWN":"UNVERIFIED",
           "MOSTLY TRUE":"MOSTLY_TRUE","MOSTLY FALSE":"MOSTLY_FALSE","HALF TRUE":"MOSTLY_TRUE"}

def norm(v):
    u = v.upper().strip()
    return ALIASES.get(u, u if u in VALID else "UNVERIFIED")

def aggregate(groq_r, qwen_r, cred_r) -> dict:
    if cred_r.get("debunked"):
        return {
            "verdict": "FALSE", "confidence": 87, "status": "danger",
            "reasoning":       groq_r.get("reasoning", "") or qwen_r.get("reasoning", ""),
            "groq_reasoning":  groq_r.get("reasoning", ""),
            "qwen_reasoning":  qwen_r.get("reasoning", "") if qwen_r.get("confidence", 0) > 0 else "",
            "key_sources": groq_r.get("key_sources", []),
        }

    gv = norm(groq_r.get("verdict", "UNVERIFIED"))
    qv = norm(qwen_r.get("verdict", "UNVERIFIED"))
    cv = norm(cred_r.get("verdict_signal", "UNVERIFIED"))
    gc = groq_r.get("confidence", 50)
    qc = qwen_r.get("confidence", 50)
    cc = cred_r.get("credibility_score", 50)

    # FIX: active keys now match WEIGHTS_KAGGLE keys exactly
    active = {"groq": gc > 0, "qwen_vl_8b": qc > 0, "credibility": True}
    tw = sum(WEIGHTS_KAGGLE[k] for k, a in active.items() if a)
    aw = {k: (WEIGHTS_KAGGLE[k] / tw if active[k] else 0.0) for k in WEIGHTS_KAGGLE}

    if not active["qwen_vl_8b"]:
        logger.info("Qwen VL 8B inactive on Kaggle — weight redistributed to Groq")

    votes = {}
    # FIX: tuple key "qwen_vl_8b" matches aw dict
    for v, c, w in [(gv, gc, "groq"), (qv, qc, "qwen_vl_8b"), (cv, cc, "credibility")]:
        nv = norm(v)
        votes[nv] = votes.get(nv, 0.0) + (c / 100.0) * aw[w]

    winner = max(votes, key=votes.get)
    base_c = (votes[winner] / sum(aw.values())) * 100
    # FIX: agreement bonus uses correct key
    bonus  = 12 if len({v for k, v in [("groq", gv), ("qwen_vl_8b", qv)] if active[k]}) == 1 else 0
    conf   = min(97, int(base_c + bonus))

    # Always prefer Groq reasoning (detailed, 35% weight) — Qwen fallback only if Groq absent
    groq_reasoning = groq_r.get("reasoning", "")
    qwen_reasoning = qwen_r.get("reasoning", "") if qc > 0 else ""
    reasoning      = groq_reasoning or qwen_reasoning

    return {
        "verdict":        winner,
        "confidence":     conf,
        "status":         VERDICT_STATUS.get(winner, "unknown"),
        "reasoning":      reasoning,
        "groq_reasoning": groq_reasoning,
        "qwen_reasoning": qwen_reasoning,
        "key_sources":    groq_r.get("key_sources", []),
        "ensemble_breakdown": {
            "groq":       {"verdict": gv, "confidence": gc, "weight": round(aw["groq"], 2)},
            # FIX: key is "qwen_vl_8b" — matches what local aggregator.py and script.js expect
            "qwen_vl_8b": {"verdict": qv, "confidence": qc,
                           "weight": round(aw["qwen_vl_8b"], 2), "available": qc > 0},
            "credibility": {"credibility_score": cc, "trusted_count": cred_r.get("trusted_count"),
                            "debunked": cred_r.get("debunked"), "weight": round(aw["credibility"], 2)},
        },
    }


# ─── Cell 9: Scraper + Search ────────────────────────────────
def scrape_content(url: str) -> dict:
    try:
        r = req_lib.get(f"https://r.jina.ai/{url}", timeout=120)
        if r.status_code == 200 and len(r.text) > 200:
            text  = r.text
            title = text.split("\n")[0].replace("#", "").strip() or "Article"
            return {"title": title, "text": text[:6000]}
    except Exception:
        pass
    try:
        scraper = cloudscraper.create_scraper()
        r = scraper.get(url, timeout=120)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}"}
        soup = BeautifulSoup(r.content, "html.parser")
        for t in soup(["script","style","nav","footer","header","aside"]):
            t.extract()
        title = (soup.title.string or "").strip() if soup.title else "Unknown"
        return {"title": title, "text": soup.get_text(separator=" ", strip=True)[:6000]}
    except Exception as e:
        return {"error": f"Scrape failed: {e}"}

def generate_query(content: dict) -> str:
    client = Groq(api_key=GROQ_API_KEY)
    try:
        r = client.chat.completions.create(
            messages=[{"role":"user","content":
                f"Generate a precise Google Search query (under 10 words) to verify this Indian news.\n"
                f"TEXT: {content['text'][:500]}\n"
                f"Expand abbreviations (BMC, CM etc). Output JSON: {{\"query\":\"...\"}}"
            }],
            model=GROQ_MODEL_NAME, response_format={"type":"json_object"},
            max_tokens=60, temperature=0.1,
        )
        return json.loads(r.choices[0].message.content)["query"]
    except Exception:
        return content.get("title", "news verification India")

def search_evidence(query: str) -> list:
    try:
        r = req_lib.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 8, "gl": "in"}, timeout=10,
        )
        return [{"title":i.get("title",""),"link":i.get("link",""),"snippet":i.get("snippet","")}
                for i in r.json().get("organic", [])]
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []


# ─── Cell 10: Initialise models ─────────────────────────────
print("Loading Qwen3-VL-8B (T4 4-bit float16)...")
qwen_model = QwenAdapter()

print("Initialising Groq client...")
groq_model = GroqReasoner()

# BGE Reranker v2-m3 — ~1.1 GB float16, fits comfortably alongside Qwen 4-bit
print("Loading BGE Reranker v2-m3 (float16)...")
from transformers import AutoModelForSequenceClassification, AutoTokenizer as _AutoTok
import torch as _torch

_BGE_MODEL_ID = "BAAI/bge-reranker-v2-m3"
_bge_tokenizer = None
_bge_model     = None
_bge_available = False

try:
    _bge_tokenizer = _AutoTok.from_pretrained(_BGE_MODEL_ID)
    _bge_model     = AutoModelForSequenceClassification.from_pretrained(
        _BGE_MODEL_ID,
        torch_dtype = _torch.float16 if _torch.cuda.is_available() else _torch.float32,
    )
    if _torch.cuda.is_available():
        _bge_model = _bge_model.cuda()
    _bge_model.eval()
    _bge_available = True
    print("BGE Reranker ready on GPU")
except Exception as _e:
    print(f"BGE Reranker failed to load: {_e}")

print(f"\nModel status: qwen_available={qwen_model.available} bge_available={_bge_available}")
print("All components ready\n")


# ─── Cell 10b: Quick sanity test (run before starting Flask) ─
import requests as _test_req, json as _test_json

def test_qwen_locally():
    """Call qwen_model.reason() directly — no Flask needed."""
    result = qwen_model.reason(
        article_text = "India won the ICC Cricket World Cup in 2011 by defeating Sri Lanka in the final.",
        evidence     = [{"title": "2011 ICC Cricket World Cup",
                         "snippet": "India defeated Sri Lanka to win the 2011 World Cup."}],
        image_url    = None,
    )
    print("Direct model test:")
    print(_test_json.dumps(result, indent=2))
    assert result["confidence"] > 0, "Model returned confidence=0 — check loading logs above"
    print("\n✓ Model working correctly\n")

test_qwen_locally()


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
        "bge_reranker": _bge_available,
        "groq":        True,
    })


_RERANK_CONFIRM_KW = [
    "confirmed", "official", "government confirms", "verified", "true",
    "authenticated", "approved", "announced", "stated", "according to",
    "pib", "ndtv confirms", "report confirms",
]
_RERANK_DENY_KW = [
    "fake", "false", "hoax", "misinformation", "fabricated", "debunked",
    "no evidence", "misleading", "unverified claim", "fact check",
    "not true", "wrong", "incorrect", "did not", "never said",
]


@ml_app.route("/rerank", methods=["POST"])
def ml_rerank():
    """
    Called by KaggleBGEReasoner in backend/ensemble/kaggle_bge.py.
    Receives pre-scraped text + evidence from local app, runs BGE inference.
    """
    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({"error": "No JSON body"}), 400

    text     = body.get("text", "")
    evidence = body.get("evidence", [])

    if not _bge_available:
        return jsonify({"verdict": "UNVERIFIED", "confidence": 0,
                        "source": "deberta_classifier", "error": "BGE not loaded"})

    if not text or not evidence:
        return jsonify({"verdict": "UNVERIFIED", "confidence": 0,
                        "source": "deberta_classifier"})

    try:
        import torch as _t
        claim = text[:512]
        pairs, ev_texts = [], []
        for ev in evidence[:6]:
            ev_text = f"{ev.get('title', '')} {ev.get('snippet', '')}".strip()
            if ev_text:
                pairs.append([claim, ev_text[:256]])
                ev_texts.append(ev_text.lower())

        if not pairs:
            return jsonify({"verdict": "UNVERIFIED", "confidence": 0,
                            "source": "deberta_classifier"})

        with _t.no_grad():
            enc = _bge_tokenizer(
                pairs, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            )
            if _t.cuda.is_available():
                enc = {k: v.cuda() for k, v in enc.items()}
            logits = _bge_model(**enc).logits.squeeze(-1)
            rel = _t.sigmoid(logits).cpu().tolist()
            if isinstance(rel, float):
                rel = [rel]

        confirm_hits = sum(1 for t in ev_texts if any(k in t for k in _RERANK_CONFIRM_KW))
        deny_hits    = sum(1 for t in ev_texts if any(k in t for k in _RERANK_DENY_KW))
        avg_rel = sum(rel) / len(rel)
        max_rel = max(rel)

        # Relevance guard + stricter deny thresholds.
        # BGE scores relevance, not truth. Indian news snippets fire deny keywords
        # on confirmatory phrases ("India rejects false claims", "factually incorrect:
        # govt denies X"). Require higher avg_rel and more deny hits for negative verdicts.
        if avg_rel < 0.40:
            verdict, confidence = "UNVERIFIED",   max(0, int(avg_rel * 50))
        elif deny_hits >= 3 and avg_rel > 0.55:
            verdict, confidence = "FALSE",        min(80, int(avg_rel * 100) + deny_hits * 3)
        elif deny_hits >= 2 and avg_rel > 0.60:
            verdict, confidence = "MOSTLY_FALSE", min(68, int(avg_rel * 90))
        elif avg_rel >= 0.72 and confirm_hits >= 2:
            verdict, confidence = "VERIFIED",     min(80, int(avg_rel * 100) + confirm_hits * 2)
        elif avg_rel >= 0.58 and (confirm_hits >= 1 or max_rel >= 0.82):
            verdict, confidence = "MOSTLY_TRUE",  min(72, int(avg_rel * 95))
        elif avg_rel >= 0.45:
            verdict, confidence = "MOSTLY_TRUE",  min(55, int(avg_rel * 75))
        else:
            verdict, confidence = "UNVERIFIED",   max(0, int(avg_rel * 60))

        logger.info(f"/rerank | avg_rel={avg_rel:.3f} confirm={confirm_hits} deny={deny_hits} → {verdict} ({confidence})")
        return jsonify({"verdict": verdict, "confidence": confidence,
                        "source": "deberta_classifier"})

    except Exception as e:
        logger.error(f"/rerank error: {e}")
        return jsonify({"verdict": "UNVERIFIED", "confidence": 0,
                        "source": "deberta_classifier", "error": str(e)})


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

    # Reject known logo/placeholder URLs so Qwen only processes real article images
    _BAD_IMAGE = ("news.google.com", "google.com/s2", "placehold.co",
                  "placeholder", "/logo", "logo.", "icon.", "favicon")
    if image_url and any(b in image_url.lower() for b in _BAD_IMAGE):
        logger.info(f"/reason: ignoring non-article image_url: {image_url[:80]}")
        image_url = None

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
        "groq_reasoning":     final.get("groq_reasoning", ""),
        "qwen_reasoning":     final.get("qwen_reasoning", ""),
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


# ─── Cell 12: Start server + ngrok ──────────────────────────
import time
from pyngrok import ngrok, conf

conf.get_default().auth_token = NGROK_AUTH_TOKEN

server_thread = threading.Thread(
    target=lambda: ml_app.run(host="0.0.0.0", port=5001, use_reloader=False)
)
server_thread.daemon = True
server_thread.start()
time.sleep(2)

tunnel     = ngrok.connect(5001, "http")
public_url = tunnel.public_url

print("\n" + "="*60)
print(f"  ML backend LIVE: {public_url}")
print("="*60)
print(f"  ➡  Set KAGGLE_BACKEND_URL = {public_url}")
print("  ➡  Paste into Render env vars or local .env")
print("="*60 + "\n")

# Keep alive
while True:
    time.sleep(60)
    logger.info(f"ML backend alive | ngrok: {public_url} | qwen={qwen_model.available}")