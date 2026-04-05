"""
ensemble/classifier.py
DeBERTa-v3-large fast classifier.

Replaces the old Qwen 3B LocalClassifier.
~50ms inference vs ~2-3s for Qwen 3B.
Loads calibrated temperature from temperature.json if present.

Ensemble weight: 30%
"""
import json
import logging
import os
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

VALID_VERDICTS = {"VERIFIED", "MOSTLY_TRUE", "MOSTLY_FALSE", "FALSE", "UNVERIFIED"}


class LocalClassifier:
    """
    DeBERTa-v3-large sequence classifier fine-tuned on LIAR + FEVER
    + FakeNewsNet + Indian datasets (3-stage curriculum).

    Loads a calibrated temperature scalar from temperature.json
    (saved during training) and applies it to soften overconfident scores.
    """

    def __init__(
        self,
        model_id: str = "prathameshbandal/sachbol-deberta-classifier",
        temperature_path: str = "./sachbol-deberta-final/temperature.json",
        device: str = "auto",
    ):
        self.model_id  = model_id
        self.available = False
        self.pipe      = None
        self.temperature = 1.0
        self._load(device, temperature_path)

    def _load(self, device: str, temperature_path: str):
        try:
            device_id = 0 if torch.cuda.is_available() else -1
            self.pipe = pipeline(
                "text-classification",
                model     = self.model_id,
                tokenizer = self.model_id,
                device    = device_id,
                truncation = True,
                max_length = 256,
                top_k      = None,   # return all class scores
            )
            self.available = True
            logger.info(f"LocalClassifier (DeBERTa) loaded from {self.model_id}")
        except Exception as exc:
            logger.warning(f"LocalClassifier unavailable: {exc}")
            return

        # Load temperature calibration
        if os.path.exists(temperature_path):
            try:
                with open(temperature_path) as f:
                    self.temperature = float(json.load(f).get("temperature", 1.0))
                logger.info(f"Temperature calibration loaded: T={self.temperature:.4f}")
            except Exception as exc:
                logger.warning(f"Could not load temperature.json: {exc} — using T=1.0")
        else:
            logger.info("temperature.json not found — using T=1.0 (no calibration)")

    def classify(self, text: str, evidence: list | None = None) -> dict:
        """
        Returns:
          verdict    : str  — one of the 5 verdict classes
          confidence : int  — 0-100 (temperature-calibrated)
          source     : str
        Note: `evidence` is accepted for interface compatibility with BGERankerClassifier
        but is not used by DeBERTa (it classifies from text alone).
        """
        if not self.available:
            return self._fallback()

        try:
            # pipeline returns list of {label, score} for all classes
            results = self.pipe(text[:512])
            if isinstance(results[0], list):
                results = results[0]

            # Apply temperature scaling to logits before softmax
            # pipeline gives softmax scores → convert back to logits,
            # apply T, re-softmax
            scores = {r["label"]: r["score"] for r in results}
            import math
            logits = {k: math.log(max(v, 1e-9)) for k, v in scores.items()}
            scaled = {k: v / self.temperature for k, v in logits.items()}
            max_l  = max(scaled.values())
            exp_s  = {k: math.exp(v - max_l) for k, v in scaled.items()}
            total  = sum(exp_s.values())
            calibrated = {k: v / total for k, v in exp_s.items()}

            best_label = max(calibrated, key=calibrated.get)
            best_score = calibrated[best_label]

            # Normalise label to valid verdict
            verdict = best_label.upper().strip()
            if verdict not in VALID_VERDICTS:
                verdict = "UNVERIFIED"

            return {
                "verdict":    verdict,
                "confidence": max(0, min(100, int(best_score * 100))),
                "source":     "deberta_classifier",
            }

        except Exception as exc:
            logger.error(f"LocalClassifier inference error: {exc}")
            return self._fallback()

    @staticmethod
    def _fallback() -> dict:
        return {"verdict": "UNVERIFIED", "confidence": 0, "source": "deberta_classifier"}
