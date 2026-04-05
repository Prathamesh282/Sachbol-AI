"""
ensemble/credibility.py
Source credibility scorer — heuristic-based, no external API needed.

Scores evidence sources based on:
  1. Domain reputation (tiered lookup table with Indian + international sources)
  2. Presence of fact-checking sites in results
  3. Debunking language in titles/snippets

Weight in ensemble: 15%
"""

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ─── Reputation tiers (domain → trust score 0-100) ────────────────────────────

DOMAIN_SCORES: dict[str, int] = {
    # Tier 1 — Government / International wire services (90–95)
    "pib.gov.in":               95,
    "mygov.in":                 92,
    "mohfw.gov.in":             92,
    "rbi.org.in":               95,
    "sebi.gov.in":              93,
    "who.int":                  95,
    "un.org":                   92,
    "reuters.com":              95,
    "apnews.com":               95,
    "afp.com":                  92,
    "bbc.com":                  93,
    "bbc.co.uk":                93,

    # Tier 2 — Established Indian news (78–88)
    "thehindu.com":             87,
    "indianexpress.com":        85,
    "ndtv.com":                 84,
    "hindustantimes.com":       82,
    "livemint.com":             82,
    "business-standard.com":   80,
    "timesofindia.indiatimes.com": 80,
    "telegraphindia.com":       80,
    "deccanherald.com":         78,
    "scroll.in":                76,
    "thewire.in":               74,
    "firstpost.com":            73,
    "news18.com":               72,
    "pti.in":                   88,
    "ani.com":                  82,

    # Tier 2 — Established international (78–90)
    "economist.com":            90,
    "ft.com":                   88,
    "bloomberg.com":            86,
    "theguardian.com":          83,
    "nytimes.com":              82,
    "washingtonpost.com":       82,
    "aljazeera.com":            79,
    "cnbc.com":                 78,

    # Tier 3 — Mixed reliability (45–65)
    "indiatvnews.com":          60,
    "abplive.com":              62,
    "zeenews.india.com":        62,
    "aajtak.in":                63,
    "tv9bharatvarsh.com":       58,
    "oneindia.com":             55,
    "jagran.com":               65,
    "bhaskar.com":              64,
    "amarujala.com":            63,

    # Tier 4 — Low credibility / known bias (20–40)
    "opindia.com":              38,
    "postcard.news":            22,
    "sudarshannews.com":        28,
    "indiatv.in":               42,
    "newsbharati.com":          38,
}

# Sites dedicated to fact-checking — special handling
FACT_CHECK_DOMAINS: set[str] = {
    "factcheck.org",
    "snopes.com",
    "boomlive.in",
    "factchecker.in",
    "altnews.in",
    "vishvasnews.com",
    "newschecker.in",
    "logically.ai",
    "thequint.com",      # Has dedicated fact-check vertical
    "indiacheck.in",
}

# Instead of generic "false", use structured markers
DEBUNK_KEYWORDS = [
    "fact check: false", "verdict: false", "claim is false", 
    "fake news", "misleading claim", "debunked:"
]

# Keywords that signal corroboration
CONFIRM_KEYWORDS = [
    "confirmed", "government confirms", "official statement",
    "ministry announces", "verified", "fact check: true",
]


class CredibilityScorer:
    """
    Analyzes the credibility of evidence sources.
    No external calls — pure domain lookup + text heuristics.
    """

    def score(self, evidence: list[dict]) -> dict:
        """
        Returns:
          credibility_score : int 0-100, average source trust
          verdict_signal    : str — directional hint for aggregator
          trusted_count     : int — number of high-trust sources (score >= 75)
          factcheck_found   : bool
          debunked          : bool — hard signal, triggers FALSE override in aggregator
          confirmed         : bool — corroboration signals found
          source            : "credibility_scorer"
        """
        if not evidence:
            return self._empty_result()

        scores: list[int] = []
        debunked     = False
        confirmed    = False
        factcheck_found = False
        trusted_count = 0

        for item in evidence:
            url     = item.get("link", "")
            title   = item.get("title", "").lower()
            snippet = item.get("snippet", "").lower()
            combined_text = title + " " + snippet
            domain  = self._domain(url)

            # Debunking signal check
            if any(kw in combined_text for kw in DEBUNK_KEYWORDS):
                debunked = True

            # Corroboration signal check
            if any(kw in combined_text for kw in CONFIRM_KEYWORDS):
                confirmed = True

            # Fact-check site detection
            if domain in FACT_CHECK_DOMAINS and any(kw in combined_text for kw in DEBUNK_KEYWORDS):
                debunked = True
            else:
                site_score = DOMAIN_SCORES.get(domain, 50)

            if site_score >= 75:
                trusted_count += 1

            scores.append(site_score)

        avg_score = int(sum(scores) / len(scores)) if scores else 50

        verdict_signal = self._compute_signal(
            avg_score, trusted_count, debunked, confirmed, factcheck_found
        )

        return {
            "credibility_score": avg_score,
            "verdict_signal":    verdict_signal,
            "trusted_count":     trusted_count,
            "factcheck_found":   factcheck_found,
            "debunked":          debunked,
            "confirmed":         confirmed,
            "source":            "credibility_scorer",
        }

    @staticmethod
    def _compute_signal(
        avg_score: int,
        trusted_count: int,
        debunked: bool,
        confirmed: bool,
        factcheck_found: bool,
    ) -> str:
        confidence_weight = 1.0
        if debunked:
            return "LOW_TRUST_SIGNAL", 0.3
        if confirmed:
            return "HIGH_TRUST_SIGNAL", 1.2
        return "NEUTRAL", 1.0
        if factcheck_found and confirmed:
            return "VERIFIED"
        if trusted_count >= 3 or (confirmed and avg_score >= 75):
            return "VERIFIED"
        if trusted_count >= 1 and avg_score >= 65:
            return "MOSTLY_TRUE"
        if avg_score < 45:
            return "MOSTLY_FALSE"
        return "UNVERIFIED"

    @staticmethod
    def _domain(url: str) -> str:
        try:
            netloc = urlparse(url).netloc.lower()
            return netloc.removeprefix("www.")
        except Exception:
            return ""

    @staticmethod
    def _empty_result() -> dict:
        return {
            "credibility_score": 30,
            "verdict_signal":    "UNVERIFIED",
            "trusted_count":     0,
            "factcheck_found":   False,
            "debunked":          False,
            "confirmed":         False,
            "source":            "credibility_scorer",
        }
