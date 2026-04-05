"""
ensemble/reranker.py
BGE Reranker v2-m3 — drop-in replacement for the DeBERTa classifier.

Model : BAAI/bge-reranker-v2-m3  (multilingual cross-encoder, 568M params)
Task  : Score semantic relevance of each evidence snippet against the article claim.
        Use relevance scores + confirmation/contradiction keywords to derive a verdict.

Why BGE reranker instead of DeBERTa?
  - No fine-tuning needed — the cross-encoder generalises well zero-shot
  - Multilingual (handles Hindi/English code-mix common in Indian news)
  - Relevance score directly measures "does this evidence address the claim?"
  - ~280ms per batch on CPU, ~40ms on GPU — fast enough for the ensemble

Ensemble weight: 25%  (same slot that DeBERTa held)
Source key    : "deberta_classifier"  ← kept for aggregator/breakdown compatibility
"""

import logging
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

VALID_VERDICTS = {"VERIFIED", "MOSTLY_TRUE", "MOSTLY_FALSE", "FALSE", "UNVERIFIED"}

# Keywords used to tilt verdict direction after scoring relevance
_CONFIRM_KW = [
    "confirmed", "official", "government confirms", "verified", "true",
    "authenticated", "approved", "announced", "stated", "according to",
    "pib", "ndtv confirms", "report confirms",
]
_DENY_KW = [
    "fake", "false", "hoax", "misinformation", "fabricated", "debunked",
    "no evidence", "misleading", "unverified claim", "fact check",
    "not true", "wrong", "incorrect", "did not", "never said",
]


class BGERankerClassifier:
    """
    Cross-encoder reranker that scores relevance between the article claim
    and each evidence snippet, then uses keyword signals to determine verdict.

    Interface is identical to LocalClassifier — drop-in replacement.
    The `evidence` parameter is optional for backward compat; without it
    the model falls back to UNVERIFIED with confidence=0 (same as DeBERTa
    when the classifier model wasn't available).
    """

    def __init__(
        self,
        model_id: str = "BAAI/bge-reranker-v2-m3",
        max_pairs: int = 6,
    ):
        self.model_id  = model_id
        self.max_pairs = max_pairs
        self.available = False
        self.model     = None
        self.tokenizer = None
        self._load()

    def _load(self):
        try:
            logger.info(f"Loading BGE reranker: {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model     = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.model.eval()
            self.available = True
            device = "GPU" if torch.cuda.is_available() else "CPU"
            logger.info(f"BGERankerClassifier ready on {device}")
        except Exception as exc:
            logger.warning(f"BGERankerClassifier unavailable: {exc}")

    def classify(self, article_text: str, evidence: list | None = None) -> dict:
        """
        Score relevance of each evidence snippet against the claim, then
        combine keyword signals and relevance to produce a verdict.

        Args:
          article_text : The article / claim text to verify
          evidence     : List of {title, snippet, link} dicts from Serper search

        Returns:
          verdict    : str — one of the 5 standard verdicts
          confidence : int — 0-100
          source     : "deberta_classifier"  ← aggregator key unchanged
        """
        if not self.available or not evidence:
            return self._fallback()

        try:
            claim  = article_text[:512]
            pairs  = []
            ev_texts = []

            for ev in evidence[: self.max_pairs]:
                ev_text = f"{ev.get('title', '')} {ev.get('snippet', '')}".strip()
                if ev_text:
                    pairs.append([claim, ev_text[:256]])
                    ev_texts.append(ev_text.lower())

            if not pairs:
                return self._fallback()

            # Batch inference
            with torch.no_grad():
                enc = self.tokenizer(
                    pairs,
                    padding    = True,
                    truncation = True,
                    max_length = 512,
                    return_tensors = "pt",
                )
                if torch.cuda.is_available():
                    enc = {k: v.cuda() for k, v in enc.items()}
                logits = self.model(**enc).logits.squeeze(-1)
                # sigmoid converts cross-encoder logit → 0–1 relevance probability
                relevance_scores = torch.sigmoid(logits).cpu().tolist()
                if isinstance(relevance_scores, float):
                    relevance_scores = [relevance_scores]

            # Keyword signals per evidence item
            confirm_hits = 0
            deny_hits    = 0
            for idx, ev_lower in enumerate(ev_texts):
                if any(kw in ev_lower for kw in _DENY_KW):
                    deny_hits += 1
                elif any(kw in ev_lower for kw in _CONFIRM_KW):
                    confirm_hits += 1

            avg_rel   = sum(relevance_scores) / len(relevance_scores)
            max_rel   = max(relevance_scores)
            n         = len(relevance_scores)

            logger.debug(
                f"BGEReranker | avg_rel={avg_rel:.3f} max_rel={max_rel:.3f} "
                f"confirm={confirm_hits} deny={deny_hits} n={n}"
            )

            # ── Verdict derivation ──────────────────────────────
            # IMPORTANT: BGE scores relevance, not truth.
            # Deny/confirm keywords in Indian news snippets fire easily on
            # confirmatory phrases ("India rejects false claims", "factually
            # incorrect: govt denies X"). Keep thresholds strict to avoid
            # false FALSE verdicts.
            #
            # Relevance guard: if avg_rel < 0.40, evidence barely covers the
            # claim — don't trust keyword signals at all.
            # Confidence capped at 80 max — BGE is a weak signal, not oracle.

            if avg_rel < 0.40:
                # Evidence is weakly relevant — abstain
                verdict    = "UNVERIFIED"
                confidence = max(0, int(avg_rel * 50))

            elif deny_hits >= 3 and avg_rel > 0.55:
                # Strong: majority of snippets have deny signals with high relevance
                verdict    = "FALSE"
                confidence = min(80, int(avg_rel * 100) + deny_hits * 3)

            elif deny_hits >= 2 and avg_rel > 0.60:
                # Moderate deny signal only when relevance is solid
                verdict    = "MOSTLY_FALSE"
                confidence = min(68, int(avg_rel * 90))

            elif avg_rel >= 0.72 and confirm_hits >= 2:
                verdict    = "VERIFIED"
                confidence = min(80, int(avg_rel * 100) + confirm_hits * 2)

            elif avg_rel >= 0.58 and (confirm_hits >= 1 or max_rel >= 0.82):
                verdict    = "MOSTLY_TRUE"
                confidence = min(72, int(avg_rel * 95))

            elif avg_rel >= 0.45:
                # Mid-range relevance, no strong signal either way
                verdict    = "MOSTLY_TRUE"
                confidence = min(55, int(avg_rel * 75))

            else:
                verdict    = "UNVERIFIED"
                confidence = max(0, int(avg_rel * 60))

            return {
                "verdict":    verdict,
                "confidence": confidence,
                "source":     "deberta_classifier",   # key kept for aggregator compat
            }

        except torch.cuda.OutOfMemoryError:
            logger.error("BGERankerClassifier OOM — clearing cache")
            torch.cuda.empty_cache()
            return self._fallback()
        except Exception as exc:
            logger.error(f"BGERankerClassifier inference error: {exc}")
            return self._fallback()

    @staticmethod
    def _fallback() -> dict:
        return {"verdict": "UNVERIFIED", "confidence": 0, "source": "deberta_classifier"}
