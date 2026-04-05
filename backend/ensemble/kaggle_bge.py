"""
ensemble/kaggle_bge.py
BGE Reranker v2-m3 reasoner that calls the Kaggle T4 backend via ngrok HTTP.

Replaces local BGERankerClassifier when USE_BGE_KAGGLE=true.

Expected Kaggle endpoint:  POST <BGE_KAGGLE_URL>/rerank
Request  body : {"text": "...", "evidence": [...]}
Response body : {"verdict": "...", "confidence": 0-100, "source": "deberta_classifier"}

The source key is kept as "deberta_classifier" throughout for aggregator compatibility.
BGE_KAGGLE_URL can be the same ngrok URL as QWEN_KAGGLE_URL — both endpoints
(/rerank and /reason) live on the same Kaggle Flask server.
"""
import logging
import requests

logger = logging.getLogger(__name__)

VALID_VERDICTS = {"VERIFIED", "MOSTLY_TRUE", "MOSTLY_FALSE", "FALSE", "UNVERIFIED"}

# Bypass ngrok browser interstitial on free tier
NGROK_HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "Content-Type": "application/json",
}


class KaggleBGEReasoner:
    """
    HTTP client that delegates BGE Reranker v2-m3 inference to the Kaggle T4 backend.

    Interface is identical to BGERankerClassifier — drop-in replacement.
    The 'available' attribute is optimistic; errors are handled per-request
    and the aggregator redistributes the weight on failure.
    """

    available = True   # optimistic — errors handled per-request

    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout

    def classify(self, article_text: str, evidence: list | None = None) -> dict:
        """
        POST article text + evidence to Kaggle /rerank endpoint.
        Falls back to UNVERIFIED with confidence=0 on any network error.

        Args:
          article_text : The article / claim text to verify
          evidence     : List of {title, snippet, link} dicts from Serper search

        Returns:
          verdict    : str — one of the 5 standard verdicts
          confidence : int — 0-100
          source     : "deberta_classifier"  ← aggregator key unchanged
        """
        if not evidence:
            return self._fallback("No evidence provided.")

        try:
            payload = {
                "text":     article_text[:512],
                "evidence": (evidence or [])[:6],
            }
            resp = requests.post(
                f"{self.base_url}/rerank",
                json    = payload,
                headers = NGROK_HEADERS,
                timeout = self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            verdict = str(data.get("verdict", "UNVERIFIED")).upper().strip()
            if verdict not in VALID_VERDICTS:
                verdict = "UNVERIFIED"

            return {
                "verdict":    verdict,
                "confidence": max(0, min(100, int(data.get("confidence", 0)))),
                "source":     "deberta_classifier",   # keep for aggregator compat
            }

        except requests.exceptions.Timeout:
            logger.warning("KaggleBGEReasoner: request timed out")
            return self._fallback("Kaggle BGE backend timed out.")

        except requests.exceptions.ConnectionError as exc:
            logger.warning(f"KaggleBGEReasoner: connection error — {exc}")
            return self._fallback("Kaggle BGE backend unreachable (ngrok may be down).")

        except Exception as exc:
            logger.error(f"KaggleBGEReasoner error: {exc}")
            return self._fallback(str(exc))

    @staticmethod
    def _fallback(reason: str = "") -> dict:
        logger.debug(f"KaggleBGEReasoner fallback: {reason}")
        return {"verdict": "UNVERIFIED", "confidence": 0, "source": "deberta_classifier"}
