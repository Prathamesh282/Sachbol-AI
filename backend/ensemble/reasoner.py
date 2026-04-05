"""
ensemble/reasoner.py
Qwen3-VL-8B local reasoner — multimodal evidence-grounded reasoning.

Replaces: QwenReasoner (Qwen-2.5-3B, text-only)
Ensemble weight: 25%  (bumped from 15% — 8B earns more trust)

Primary value:
  - Produces human-readable reasoning shown in the UI
  - Can analyse article images / embedded screenshots alongside text
  - Groq 70B still handles verdict accuracy (35% weight)

Usage:
  reasoner = QwenVLReasoner()
  result   = reasoner.reason(article_text, evidence, image_url="https://...")
"""

import json
import logging
import torch

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

logger = logging.getLogger(__name__)

VALID_VERDICTS = {"VERIFIED", "MOSTLY_TRUE", "MOSTLY_FALSE", "FALSE", "UNVERIFIED"}

SYSTEM_PROMPT = """You are SachBol, an expert fact-checking AI for Indian and world news.

STRICT RULES:
1. Use ONLY the evidence provided (text snippets + any image) to reach your verdict.
2. Do NOT use external knowledge not present in the evidence.
3. If evidence is insufficient, contradictory, or irrelevant → verdict must be UNVERIFIED.
4. Respond ONLY with valid JSON. No text outside the JSON block.

Verdict definitions:
- VERIFIED     : Evidence clearly supports the claim
- MOSTLY_TRUE  : Evidence partially supports with minor inaccuracies
- MOSTLY_FALSE : Evidence partially contradicts or is insufficient
- FALSE        : Evidence clearly contradicts the claim
- UNVERIFIED   : Evidence is absent, irrelevant, or too contradictory to judge"""

_PROMPT_WITH_EVIDENCE = """\
Fact-check this claim using ONLY the evidence provided.

CLAIM: {claim}

EVIDENCE:
{evidence}

Respond with JSON only:
{{"verdict": "VERIFIED|MOSTLY_TRUE|MOSTLY_FALSE|FALSE|UNVERIFIED", "confidence": <0-100>, "reasoning": "<2-3 sentences citing the evidence>"}}"""

_PROMPT_NO_EVIDENCE = """\
Fact-check this claim. No external evidence is available.

CLAIM: {claim}

Respond with JSON only:
{{"verdict": "VERIFIED|MOSTLY_TRUE|MOSTLY_FALSE|FALSE|UNVERIFIED", "confidence": <0-100>, "reasoning": "<brief explanation>"}}"""


class QwenVLReasoner:
    """
    Local Qwen3-VL-8B fine-tuned reasoner.
    Loaded once at startup, runs inference per request.
    Accepts an optional image_url for multimodal fact-checking.
    """

    def __init__(
        self,
        model_id: str       = "prathameshbandal/sachbol-qwen3vl-final",
        max_text_chars: int = 800,
    ):
        self.model_id       = model_id
        self.max_text_chars = max_text_chars
        self.available      = False
        self.model          = None
        self.processor      = None
        self._load()

    def _load(self):
        try:
            logger.info(f"Loading QwenVLReasoner (8B) from {self.model_id}...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                padding_side="right",
            )
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map        = "auto",
                torch_dtype       = torch.bfloat16,
                trust_remote_code = True,
            )
            self.model.eval()
            self.available = True
            logger.info("QwenVLReasoner (8B) ready")
        except Exception as exc:
            logger.warning(f"QwenVLReasoner unavailable: {exc}")

    def reason(
        self,
        article_text: str,
        evidence: list[dict],
        image_url: str | None = None,
    ) -> dict:
        """
        Generate a verdict + reasoning grounded in the evidence.

        Args:
          article_text : The claim / article body to verify
          evidence     : List of {title, snippet, link} dicts from Serper
          image_url    : Optional direct URL of an image from the article.
                         When present, Qwen3-VL will process it visually.

        Returns:
          verdict    : str
          confidence : int
          reasoning  : str  ← shown in the UI
          source     : "qwen_vl_8b"
          used_image : bool
        """
        if not self.available:
            return self._fallback()

        claim   = article_text[: self.max_text_chars]
        ev_text = self._format_evidence(evidence)
        prompt  = (
            _PROMPT_WITH_EVIDENCE.format(claim=claim, evidence=ev_text)
            if ev_text
            else _PROMPT_NO_EVIDENCE.format(claim=claim)
        )

        # Build message — multimodal if image_url provided, text-only otherwise
        if image_url:
            user_content = [
                {"type": "image", "image": image_url},
                {"type": "text",  "text": prompt},
            ]
        else:
            user_content = prompt

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        try:
            text_input = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            if image_url:
                # qwen-vl-utils handles image downloading + tokenisation
                try:
                    from qwen_vl_utils import process_vision_info
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.processor(
                        text   = [text_input],
                        images = image_inputs,
                        videos = video_inputs,
                        return_tensors = "pt",
                        truncation     = True,
                        max_length     = 2048,
                    ).to(self.model.device)
                except ImportError:
                    logger.warning("qwen-vl-utils not installed — falling back to text-only")
                    image_url = None
                    inputs = self.processor.tokenizer(
                        text_input, return_tensors="pt",
                        truncation=True, max_length=2048,
                    ).to(self.model.device)
            else:
                inputs = self.processor.tokenizer(
                    text_input, return_tensors="pt",
                    truncation=True, max_length=2048,
                ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens = 200,
                    do_sample      = False,
                    pad_token_id   = self.processor.tokenizer.eos_token_id,
                )

            input_len  = inputs["input_ids"].shape[1]
            new_tokens = output_ids[0][input_len:]
            raw = self.processor.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

            result = self._parse(raw)
            result["used_image"] = bool(image_url)
            return result

        except Exception as exc:
            logger.error(f"QwenVLReasoner inference error: {exc}")
            return self._fallback()

    @staticmethod
    def _format_evidence(evidence: list[dict]) -> str:
        if not evidence:
            return ""
        lines = []
        for item in evidence[:5]:
            title   = item.get("title",   "").strip()
            snippet = item.get("snippet", "").strip()
            if title:
                lines.append(f"- {title}: {snippet[:140]}" if snippet else f"- {title}")
        return "\n".join(lines)

    def _parse(self, raw: str) -> dict:
        try:
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start >= 0 and end > start:
                data    = json.loads(raw[start:end])
                verdict = str(data.get("verdict", "UNVERIFIED")).upper().strip()
                if verdict not in VALID_VERDICTS:
                    verdict = "UNVERIFIED"
                return {
                    "verdict":    verdict,
                    "confidence": max(0, min(100, int(data.get("confidence", 50)))),
                    "reasoning":  str(data.get("reasoning", ""))[:600],
                    "source":     "qwen_vl_8b",
                }
        except Exception as exc:
            logger.error(f"QwenVLReasoner parse error: {exc} | raw={raw[:100]}")
        return self._fallback()

    @staticmethod
    def _fallback() -> dict:
        return {
            "verdict":    "UNVERIFIED",
            "confidence": 0,
            "reasoning":  "Qwen VL 8B reasoner not available.",
            "source":     "qwen_vl_8b",
            "used_image": False,
        }


# ── Backward-compat alias so any code still importing QwenReasoner doesn't break
QwenReasoner = QwenVLReasoner
