"""
ensemble/kaggle_reasoner.py
Qwen3-VL-8B reasoner that calls the Kaggle T4 backend via ngrok HTTP.

Replaces: KaggleQwenReasoner (was 3B text-only, source="qwen_3b")

Expected Kaggle endpoint:  POST <QWEN_KAGGLE_URL>/reason
Request  body: {"text": "...", "evidence": [...], "image_url": "..."|null}
Response body: {"verdict": "...", "confidence": 0-100, "reasoning": "..."}

Ensemble weight: 25%  (bumped from 15%)
"""
import json
import logging
import requests

logger = logging.getLogger(__name__)

VALID_VERDICTS = {"VERIFIED", "MOSTLY_TRUE", "MOSTLY_FALSE", "FALSE", "UNVERIFIED"}

# ngrok free tier serves an HTML interstitial to non-browser clients.
# This header bypasses it so Flask on Kaggle receives the JSON body.
NGROK_HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "Content-Type": "application/json",
}


class KaggleQwenReasoner:
    """HTTP client that delegates Qwen3-VL-8B inference to the Kaggle T4 GPU backend."""

    available = True   # optimistic — errors handled per-request

    def __init__(self, base_url: str, timeout: int = 45):
        # Timeout bumped from 35 → 45s: 8B inference is slightly slower than 3B
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout

    def reason(
        self,
        article_text: str,
        evidence: list,
        image_url: str | None = None,
    ) -> dict:
        """
        POST article + evidence (+ optional image_url) to Kaggle backend.
        Falls back to UNVERIFIED with confidence=0 on any network error so the
        aggregator simply redistributes its 25% weight.
        """
        try:
            payload = {
                "text":      article_text[:800],   # 8B handles longer context
                "evidence":  evidence[:5],
                "image_url": image_url,             # None is serialised as null — Kaggle checks for it
            }
            resp = requests.post(
                f"{self.base_url}/reason",
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
                "confidence": max(0, min(100, int(data.get("confidence", 50)))),
                "reasoning":  str(data.get("reasoning", ""))[:2000],
                "source":     "qwen_vl_8b",     # ← updated from "qwen_3b"
                "used_image": bool(image_url and data.get("used_image")),
            }

        except requests.exceptions.Timeout:
            logger.warning("KaggleQwenReasoner: request timed out")
            return self._fallback("Kaggle backend timed out.")

        except requests.exceptions.ConnectionError as exc:
            logger.warning(f"KaggleQwenReasoner: connection error — {exc}")
            return self._fallback("Kaggle backend unreachable (ngrok may be down).")

        except Exception as exc:
            logger.error(f"KaggleQwenReasoner error: {exc}")
            return self._fallback(str(exc))

    @staticmethod
    def _fallback(reason: str = "") -> dict:
        return {
            "verdict":    "UNVERIFIED",
            "confidence": 0,
            "reasoning":  reason,
            "source":     "qwen_vl_8b",
            "used_image": False,
        }
