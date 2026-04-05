"""
ensemble/groq_reasoner.py
Groq Llama-3.3-70b deep reasoner — highest accuracy, highest weight.

Split from the old reasoner.py so each model has its own file.
Ensemble weight: 40%
"""
import json
import logging
from groq import Groq

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are TruthGuard, an elite fact-checking AI. Verify news claims using Google search evidence.

DECISION RULES:
- 2+ reputable sources (BBC, Reuters, NDTV, Times of India, AP, PIB, government sites) confirm core claim → VERIFIED
- Sources confirm event but key details are wrong/exaggerated → MOSTLY_TRUE
- Only low-credibility or unrelated sources found → MOSTLY_FALSE
- Fact-checkers explicitly call it hoax/fake/misinformation → FALSE
- Insufficient direct evidence but article is about a verifiable public event
→ MOSTLY_TRUE or MOSTLY_FALSE based on background context
Output ONLY valid JSON. No text outside the JSON block."""

USER_TEMPLATE = """Verify this news article using the evidence below.

ARTICLE:
{article_text}

EVIDENCE (Google search results):
{evidence_json}

Write detailed reasoning in 3-4 paragraphs:
- Para 1: What the article claims
- Para 2: What the evidence confirms or contradicts (cite sources by name)
- Para 3: Any red flags, missing context, or corroborating details
- Para 4: Your final assessment and why

Respond with this exact JSON structure:
{{
  "verdict": "VERIFIED" | "MOSTLY_TRUE" | "MOSTLY_FALSE" | "FALSE" | "UNVERIFIED",
  "confidence": <integer 0-100>,
  "reasoning": "<3-4 detailed paragraphs separated by \\n\\n>",
  "key_sources": ["<domain1>", "<domain2>"]
}}"""


class GroqReasoner:

    def __init__(self, api_key: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        self.client = Groq(api_key=api_key)
        self.model  = model

    def reason(self, article_text: str, evidence: list[dict]) -> dict:
        if not evidence:
            return self._reason_no_evidence(article_text)

        prompt = USER_TEMPLATE.format(
            article_text  = article_text[:1200],
            evidence_json = json.dumps(evidence[:6], indent=2),
        )
        try:
            resp = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                model           = self.model,
                response_format = {"type": "json_object"},
                temperature     = 0.1,
                max_tokens      = 1500,
            )
            data = json.loads(resp.choices[0].message.content)
            return {
                "verdict":     str(data.get("verdict",    "UNVERIFIED")).upper(),
                "confidence":  max(0, min(100, int(data.get("confidence", 50)))),
                "reasoning":   str(data.get("reasoning",  ""))[:2000],
                "key_sources": data.get("key_sources", []),
                "source":      "groq",
            }
        except Exception as exc:
            logger.error(f"GroqReasoner error: {exc}")
            return {"verdict": "UNVERIFIED", "confidence": 0,
                    "reasoning": f"Groq API error: {exc}",
                    "key_sources": [], "source": "groq"}

    def _reason_no_evidence(self, article_text: str) -> dict:
        try:
            prompt = (
                f"Analyze this news article for signs of misinformation. "
                f"No external evidence is available — reason from the text itself.\n\n"
                f"ARTICLE: {article_text[:800]}\n\n"
                f'Output JSON: {{"verdict":"VERIFIED|MOSTLY_TRUE|MOSTLY_FALSE|FALSE|UNVERIFIED",'
                f'"confidence":<0-60>,"reasoning":"<explanation>","key_sources":[]}}'
            )
            resp = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                model=self.model, response_format={"type": "json_object"},
                temperature=0.1, max_tokens=256,
            )
            data = json.loads(resp.choices[0].message.content)
            return {
                "verdict":     str(data.get("verdict", "UNVERIFIED")).upper(),
                "confidence":  max(0, min(60, int(data.get("confidence", 30)))),
                "reasoning":   str(data.get("reasoning", "No evidence found."))[:2000],
                "key_sources": [],
                "source":      "groq",
            }
        except Exception as exc:
            return {"verdict": "UNVERIFIED", "confidence": 0,
                    "reasoning": str(exc), "key_sources": [], "source": "groq"}
