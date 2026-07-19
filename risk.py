import json
import logging
import re
from rapidfuzz import fuzz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskDetector:
    """
    Rule-based + fuzzy risk detector.
    Works across ALL contract types (Service, License, Employment,
    NDA, SaaS, Franchise, Lease, Partnership, Purchase, etc.)
    """

    def __init__(self, risk_json_path, fuzzy_threshold=95):
        self.threshold = fuzzy_threshold

        with open(risk_json_path, "r", encoding="utf-8") as f:
            self.rules = json.load(f)

    def _extract_snippet(self, text, phrase, window=120):
        idx = text.lower().find(phrase.lower())
        if idx == -1:
            return ""
        start = max(0, idx - window)
        end = min(len(text), idx + len(phrase) + window)
        return text[start:end]

    def detect_risks(self, text, clause_type=None):
        """
        clause_type is optional. If a rule has an 'applies_to' list and
        clause_type is provided, only patterns relevant to that clause
        type (or with no restriction) are checked. This keeps detection
        accurate across different agreement types without needing
        separate detectors per contract type.
        """
        findings = []
        seen = set()

        for rule in self.rules:
            applies_to = rule.get("applies_to")
            if applies_to and clause_type and clause_type not in applies_to:
                continue

            pattern = rule["pattern"]
            risk = rule["risk"]
            level = rule["level"]

            # Exact Match
            if pattern.lower() in text.lower():
                key = (pattern, risk)
                if key not in seen:
                    findings.append(
                        {
                            "pattern": pattern,
                            "risk": risk,
                            "level": level,
                            "match_type": "exact",
                            "matched_text": self._extract_snippet(
                                text,
                                pattern
                            ),
                        }
                    )
                    seen.add(key)
                continue

            # Fuzzy Match
            sentences = re.split(r"[.\n]", text)

            for sentence in sentences:
                score = fuzz.partial_ratio(
                    pattern.lower(),
                    sentence.lower()
                )

                if score >= self.threshold:
                    key = (pattern, risk)
                    if key not in seen:
                        findings.append(
                            {
                                "pattern": pattern,
                                "risk": risk,
                                "level": level,
                                "match_type": "fuzzy",
                                "matched_text": sentence.strip(),
                            }
                        )
                        seen.add(key)
                    break

        return findings
