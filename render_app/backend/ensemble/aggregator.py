"""
ensemble/aggregator.py
Weighted voting across all 4 ensemble components.

Weights (v3 — BGE demoted, Qwen promoted):
  Groq LLaMA Scout → 40%  (primary reasoner — detailed, citation-grounded)
  Qwen VL 8B       → 30%  (evidence-grounded multimodal reasoning — promoted)
  BGE Reranker     → 15%  (relevance signal only — demoted; prone to keyword false-positives)
  Credibility      → 15%  (source reputation heuristics — unchanged)

BGE is demoted because it is a retrieval-relevance model (MS MARCO), not a
fact-checking model. Its deny-keyword logic fires on Indian news snippets that
contain words like "rejects", "incorrect", "dismissing" in a confirmatory
context (e.g. "India rejects false payment claims"). Qwen reasons over the
full claim + evidence and is more trustworthy for verdict direction.

Hard overrides (applied before voting):
  1. Credibility DEBUNKED signal → force FALSE (when LLM confidence < 85)
  2. All available models agree  → confidence +12 bonus
"""
import logging

logger = logging.getLogger(__name__)

WEIGHTS = {
    "groq":               0.40,   # primary reasoner — up from 0.35
    "deberta_classifier": 0.15,   # BGE relevance signal — demoted from 0.25
    "qwen_vl_8b":         0.30,   # evidence-grounded reasoning — up from 0.25
    "credibility_scorer": 0.15,
}

VERDICT_TO_STATUS = {
    "VERIFIED":     "safe",
    "MOSTLY_TRUE":  "caution",
    "MOSTLY_FALSE": "caution",
    "FALSE":        "danger",
    "UNVERIFIED":   "unknown",
}

VERDICT_ALIASES = {
    "TRUE":         "VERIFIED",
    "REAL":         "VERIFIED",
    "MOSTLY TRUE":  "MOSTLY_TRUE",
    "HALF TRUE":    "MOSTLY_TRUE",
    "MOSTLY FALSE": "MOSTLY_FALSE",
    "FAKE":         "FALSE",
    "UNKNOWN":      "UNVERIFIED",
}

VALID_VERDICTS = {"VERIFIED", "MOSTLY_TRUE", "MOSTLY_FALSE", "FALSE", "UNVERIFIED"}


def _norm(verdict: str) -> str:
    v = verdict.upper().strip()
    return VERDICT_ALIASES.get(v, v if v in VALID_VERDICTS else "UNVERIFIED")


class EnsembleAggregator:

    def aggregate(
        self,
        groq_result:        dict,
        deberta_result:     dict,
        qwen_result:        dict,
        credibility_result: dict,
    ) -> dict:

        groq_conf   = groq_result.get("confidence", 0)
        is_debunked = credibility_result.get("debunked")

        # ── Hard override: debunked ────────────────────────────
        if is_debunked and groq_conf < 85:
            logger.info("DEBUNKED override triggered (LLM confidence below threshold)")
            reasoning = self._pick_reasoning(qwen_result, groq_result)
            return {
                "verdict":    "FALSE",
                "confidence": 87,
                "status":     "danger",
                "reasoning":  reasoning + " [Fact-check signals detected conflict with report.]",
                "groq_reasoning": groq_result.get("reasoning", ""),
                "qwen_reasoning": qwen_result.get("reasoning", "") if qwen_result.get("confidence", 0) > 0 else "",
                "key_sources": groq_result.get("key_sources", []),
                "ensemble_breakdown": self._breakdown(
                    groq_result, deberta_result, qwen_result, credibility_result
                ),
            }

        # ── Normalize all verdicts ─────────────────────────────
        groq_v    = _norm(groq_result.get("verdict",    "UNVERIFIED"))
        deberta_v = _norm(deberta_result.get("verdict", "UNVERIFIED"))
        qwen_v    = _norm(qwen_result.get("verdict",    "UNVERIFIED"))
        raw_signal = credibility_result.get("verdict_signal", "UNVERIFIED")
        if isinstance(raw_signal, tuple):
            cred_v = _norm(raw_signal[0])
        else:
            cred_v = _norm(raw_signal)

        groq_conf    = groq_result.get("confidence",    50)
        deberta_conf = deberta_result.get("confidence", 50)
        qwen_conf    = qwen_result.get("confidence",    50)
        cred_score   = credibility_result.get("credibility_score", 50)

        # ── Identify which models are active ──────────────────
        active = {
            "groq":               groq_conf    > 0,
            "deberta_classifier": deberta_conf > 0,
            "qwen_vl_8b":         qwen_conf    > 0,   # key updated
            "credibility_scorer": True,
        }

        # Renormalize weights for active models only
        total_w = sum(WEIGHTS[k] for k, a in active.items() if a)
        aw = {k: (WEIGHTS[k] / total_w if active[k] else 0.0) for k in WEIGHTS}

        if not active["deberta_classifier"]:
            logger.info("DeBERTa inactive — weight redistributed")
        if not active["qwen_vl_8b"]:
            logger.info("Qwen VL 8B inactive — weight redistributed")

        # ── Weighted vote ──────────────────────────────────────
        votes: dict[str, float] = {}

        def cast(verdict: str, conf: int, weight: float):
            v = _norm(verdict)
            score = (conf / 100.0) * weight
            # Penalize VERIFIED votes when debunk signal present
            if v == "VERIFIED" and credibility_result.get("debunked"):
                score *= 0.5
            votes[v] = votes.get(v, 0.0) + score

        cast(groq_v,    groq_conf,    aw["groq"])
        cast(deberta_v, deberta_conf, aw["deberta_classifier"])
        cast(qwen_v,    qwen_conf,    aw["qwen_vl_8b"])        # key updated
        cast(cred_v,    cred_score,   aw["credibility_scorer"])

        final_verdict  = max(votes, key=votes.get)
        winning_score  = votes[final_verdict]
        total_active_w = sum(aw.values())
        base_conf      = (winning_score / total_active_w) * 100

        # Agreement bonus
        active_verdicts = [
            v for k, v in [
                ("groq",               groq_v),
                ("deberta_classifier", deberta_v),
                ("qwen_vl_8b",         qwen_v),        # key updated
                ("credibility_scorer", cred_v),
            ] if active.get(k, True)
        ]
        bonus            = 12 if len(set(active_verdicts)) == 1 else 0
        final_confidence = min(97, int(base_conf + bonus))

        # ── Pick best reasoning for UI ─────────────────────────
        # Groq 70B preferred — detailed analysis, 35% weight.
        # Qwen VL 8B is the fallback (and preferred when image was used).
        reasoning = self._pick_reasoning(qwen_result, groq_result)
        groq_reasoning = groq_result.get("reasoning", "")
        qwen_reasoning = qwen_result.get("reasoning", "") if qwen_result.get("confidence", 0) > 0 else ""

        return {
            "verdict":            final_verdict,
            "confidence":         final_confidence,
            "status":             VERDICT_TO_STATUS.get(final_verdict, "unknown"),
            "reasoning":          reasoning,
            "groq_reasoning":     groq_reasoning,
            "qwen_reasoning":     qwen_reasoning,
            "key_sources":        groq_result.get("key_sources", []),
            "ensemble_breakdown": self._breakdown(
                groq_result, deberta_result, qwen_result, credibility_result, aw
            ),
        }

    @staticmethod
    def _pick_reasoning(qwen: dict, groq: dict) -> str:
        """
        Prefer Groq reasoning (detailed, 35% weight).
        If Qwen used an image, prefer Qwen reasoning — it has visual context Groq lacks.
        Falls back to Groq if Qwen unavailable.
        """
        if qwen.get("used_image") and qwen.get("confidence", 0) > 0:
            return qwen.get("reasoning", "")
        gr = groq.get("reasoning", "")
        if gr and groq.get("confidence", 0) > 0:
            return gr
        return qwen.get("reasoning", "")

    @staticmethod
    def _breakdown(groq, deberta, qwen, credibility, aw=None) -> dict:
        w = aw or WEIGHTS
        return {
            "groq": {
                "verdict":    groq.get("verdict"),
                "confidence": groq.get("confidence"),
                "weight":     round(w["groq"], 2),
            },
            "deberta": {
                "verdict":    deberta.get("verdict"),
                "confidence": deberta.get("confidence"),
                "weight":     round(w["deberta_classifier"], 2),
                "available":  deberta.get("confidence", 0) > 0,
            },
            "qwen_vl_8b": {                              # key updated from "qwen_3b"
                "verdict":    qwen.get("verdict"),
                "confidence": qwen.get("confidence"),
                "weight":     round(w["qwen_vl_8b"], 2),
                "available":  qwen.get("confidence", 0) > 0,
                "used_image": qwen.get("used_image", False),
            },
            "credibility": {
                "credibility_score": credibility.get("credibility_score"),
                "trusted_count":     credibility.get("trusted_count"),
                "debunked":          credibility.get("debunked"),
                "weight":            round(w["credibility_scorer"], 2),
            },
        }
