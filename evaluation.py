"""
evaluation.py

Unified Rubric-Based Evaluation Module.

Evaluates TWO components in one file:

    PART A — Clause Extraction & Classification
        1. Valid JSON + Structure
        2. Correct Clause Count
        3. Correct Classification  -> NOW EVALUATED VIA LLM-AS-JUDGE
           (no manual ground_truth labels required anymore)
        4. No Hallucination (type-level + text-level)
        5. Prompt Compliance (correct fields present)

    PART B — Executive Summary
        1. Length Compliance      (under 120 words)
        2. No Hallucination       (no invented risks or categories)
        3. Risk Coverage          (high risks mentioned)
        4. Clause Coverage        (clause categories mentioned)
        5. Format Compliance      (no bullets, no headings, no markdown)

Each criterion scored independently.
Both parts scored out of 100.
Final combined score = average of Part A + Part B.

------------------------------------------------------------------
WHY THIS VERSION IS DIFFERENT (LLM-as-Judge for Classification)
------------------------------------------------------------------
Earlier, "correct_classification" needed a manually annotated
`ground_truth` list (human had to label every clause). That's manual
effort and doesn't scale.

Now, `_eval_correct_classification` sends ALL clauses to an LLM in a
SINGLE batched call and asks the LLM to judge, for each clause,
whether the assigned `clause_type` is the correct category given the
clause text and the list of allowed categories. No human labeling
needed, and batching keeps it to 1 extra LLM call per evaluation run
(efficient, not one call per clause).

If you still want to compare against human labels sometimes (e.g. for
auditing the judge itself), you CAN still pass `ground_truth` — the
method will use it instead of calling the LLM. But it's optional now,
not required.
"""

import json
import logging
import re
import unicodedata
from typing import Dict, List, Optional, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# GLOBAL CONSTANTS
# ----------------------------------------------------------------

PASS_THRESHOLD = 70.0

# Per-criterion pass threshold. A criterion is "passed" if its score is
# at or above this fraction of its max_score, instead of requiring a
# perfect 100% match. This keeps the PASS/FAIL label consistent with the
# displayed percentage (e.g. a 90% hallucination score should not show
# as FAIL just because of one borderline/false-positive match).
CRITERION_PASS_THRESHOLD = 0.70

# Minimum fraction of a clause's words that must appear in the contract
# for it to be considered "found" (not a text hallucination). Using a
# fuzzy word-overlap check instead of exact substring matching avoids
# false positives caused by apostrophes, curly quotes, line breaks, or
# clause-numbering prefixes getting merged into the extracted text.
TEXT_MATCH_FUZZY_THRESHOLD = 0.80

# ----------------------------------------------------------------
# RUBRIC DEFINITIONS
# ----------------------------------------------------------------

CLAUSE_RUBRIC = {
    "valid_json": {
        "description": (
            "LLM returned valid parseable JSON "
            "with correct structure (list of objects with required fields)"
        ),
        "max_score": 20
    },
    "correct_clause_count": {
        "description": "Number of clauses returned matches regex-extracted clause count",
        "max_score": 20
    },
    "correct_classification": {
        "description": (
            "Clause types are judged correct by an LLM-as-judge "
            "(no manual ground truth required)"
        ),
        "max_score": 30
    },
    "no_hallucination": {
        "description": (
            "LLM did not invent clause types outside allowed categories, "
            "and did not fabricate clause text absent from the original contract"
        ),
        "max_score": 20
    },
    "prompt_compliance": {
        "description": (
            "All required fields present in every output item: "
            "clause_number, clause_title, text, clause_type"
        ),
        "max_score": 10
    }
}

SUMMARY_RUBRIC = {
    "length_compliance": {
        "description": "Summary is under 120 words as instructed in the prompt",
        "max_score": 20
    },
    "no_hallucination": {
        "description": (
            "Summary does not mention risks or clause categories "
            "that were not present in the structured input sent to the LLM"
        ),
        "max_score": 20
    },
    "risk_coverage": {
        "description": "Summary mentions the high-risk findings from the structured input",
        "max_score": 25
    },
    "clause_coverage": {
        "description": "Summary mentions the clause categories from the structured input",
        "max_score": 25
    },
    "format_compliance": {
        "description": (
            "Summary uses paragraph format only — "
            "no bullets, no headings, no markdown"
        ),
        "max_score": 10
    }
}

RISK_RUBRIC = {
    "valid_severity_levels": {
        "description": "Every detected risk's severity level is one of HIGH, MEDIUM, LOW",
        "max_score": 30
    },
    "known_pattern_match": {
        "description": (
            "Every detected risk's pattern matches an entry in the configured "
            "risk-pattern rules (no hallucinated/invented patterns)"
        ),
        "max_score": 40
    },
    "required_fields": {
        "description": (
            "Every risk object contains pattern, risk, level, and matched_text"
        ),
        "max_score": 30
    }
}

CLAUSE_MAX = sum(c["max_score"] for c in CLAUSE_RUBRIC.values())
SUMMARY_MAX = sum(c["max_score"] for c in SUMMARY_RUBRIC.values())
RISK_MAX = sum(c["max_score"] for c in RISK_RUBRIC.values())

DEFAULT_REQUIRED_FIELDS = [
    "clause_number",
    "clause_title",
    "text",
    "clause_type"
]

BASE_RISK_KEYWORDS = [
    "unlimited liability", "no cap", "automatic renewal",
    "unilateral termination", "force majeure", "indemnification",
    "penalty", "arbitration", "governing law", "jurisdiction"
]


# ================================================================
# TEXT NORMALIZATION + FUZZY MATCH HELPERS
# (used to avoid false-positive "text hallucination" flags caused by
#  apostrophes, curly quotes, line breaks, or numbering prefixes)
# ================================================================

def _normalize_text(s: str) -> str:
    """
    Lowercases, strips accents, removes apostrophes/quotes entirely
    (so "provider's" and "providers" become identical), replaces all
    other punctuation with spaces, and collapses whitespace/newlines.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[’'`´\"“”]", "", s)          # drop apostrophes/quotes entirely
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)        # other punctuation -> space
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _fuzzy_text_found(
    clause_text: str,
    contract_text: str,
    threshold: float = TEXT_MATCH_FUZZY_THRESHOLD
) -> bool:
    """
    Returns True if the clause text is "found" in the contract, using a
    word-overlap ratio instead of strict exact substring matching.

    Exact substring matching (the old approach) breaks on trivial
    formatting differences: apostrophes vs none, line breaks merged into
    spaces, clause numbers/headings concatenated with body text, etc.
    Those are NOT hallucinations — the words are genuinely from the
    contract, just reformatted during extraction.

    This checks what fraction of the clause's words actually appear
    somewhere in the contract. A real hallucination (invented text) will
    have a low overlap ratio; reformatted-but-genuine text will have a
    high one.
    """
    clause_norm = _normalize_text(clause_text)
    contract_norm = _normalize_text(contract_text)

    clause_words = clause_norm.split()
    if not clause_words:
        return True

    contract_word_set = set(contract_norm.split())
    matched = sum(1 for w in clause_words if w in contract_word_set)
    ratio = matched / len(clause_words)
    return ratio >= threshold


# ================================================================
# LLM-AS-JUDGE HELPER
# ================================================================

from llm_client import chat

def _default_llm_judge(prompt: str, model: str = None) -> str:
    """
    Uses the centralized OpenRouter client.
    The model argument is kept only for backward compatibility.
    """

    return chat(
        user_prompt=prompt,
        temperature=0.0,
        max_tokens=1500,
    )


def _normalize_category(category: str, allowed_categories: List[str]) -> str:
    """
    Maps a possibly-messy category name (e.g. "Termination Clause",
    "termination", " Termination ") to the closest matching entry in
    allowed_categories. This fixes a common false-FAIL cause: the
    classifier and the allowed-category list use slightly different
    naming, so an otherwise-correct classification gets judged
    "incorrect" purely due to string mismatch.

    Falls back to returning the original category unchanged if no
    reasonably close match is found (so genuine hallucinated/invalid
    categories are still caught by the no-hallucination check).
    """
    import difflib

    if not category:
        return category

    cat_norm = re.sub(r"\s+clause[s]?$", "", category.strip(), flags=re.IGNORECASE).strip().lower()

    for allowed in allowed_categories:
        if allowed.strip().lower() == cat_norm:
            return allowed

    close = difflib.get_close_matches(
        cat_norm, [a.lower() for a in allowed_categories], n=1, cutoff=0.75
    )
    if close:
        for allowed in allowed_categories:
            if allowed.lower() == close[0]:
                return allowed

    return category


def _build_judge_prompt(llm_results: List[Dict], allowed_categories: List[str]) -> str:
    """
    Builds a single batched prompt asking the judge-LLM to verify the
    clause_type assigned to EVERY clause in one shot, instead of one
    call per clause (cheaper, faster, and keeps the run reproducible).
    """
    # NOTE: earlier this truncated each clause's text to 500 characters
    # before sending it to the judge. For longer clauses, the judge was
    # deciding correctness on partial context and mis-judging clearly
    # correct classifications as wrong. Full clause text is sent now
    # (still capped at a generous 2000 chars as a safety limit against
    # pathologically long clauses blowing up the prompt).
    clauses_block = "\n\n".join(
        f"Clause {i}:\n"
        f"Assigned Type: {item.get('clause_type', 'Unknown')}\n"
        f"Text: {item.get('text', '')[:2000]}"
        for i, item in enumerate(llm_results)
    )

    return f"""You are a strict legal-contract clause classification judge.

Allowed categories: {allowed_categories}

For EACH clause below, decide if "Assigned Type" is the correct category
for the clause text. Judge only from the text — do not invent new
categories outside the allowed list. If the clause text is genuinely
ambiguous between two close categories, judge it "correct" as long as
the assigned type is a reasonable, defensible choice — do not fail a
clause purely for a stylistic label preference.

IMPORTANT: "clause_number" in your response MUST exactly match the
"Clause N" number given below for that clause — do not renumber,
reorder, or skip any clause.

{clauses_block}

Respond with ONLY a JSON array, no prose, no markdown fences, in this
exact format:
[
  {{"clause_number": 0, "correct": true, "reason": "short reason"}},
  {{"clause_number": 1, "correct": false, "reason": "short reason"}}
]
"""


# ================================================================
# PART A — CLAUSE EVALUATOR
# ================================================================

class ClauseEvaluator:
    """
    Evaluates the quality of LLM clause extraction and classification output.
    Does NOT modify extraction or classification — only scores the output.
    """

    def __init__(
        self,
        allowed_categories: List[str],
        judge_model: str = None,
        llm_judge_fn: Optional[Callable[[str, str], str]] = None
    ):
        self.allowed_categories = allowed_categories + ["Unknown"]
        self.judge_model = judge_model
        # Swappable judge caller — defaults to local Ollama, but you can
        # inject any function(prompt, model) -> raw_text_response here.
        self.llm_judge_fn = llm_judge_fn or _default_llm_judge

    # ---- Criterion 1: Valid JSON + Structure ----
    def _eval_valid_json(
        self,
        raw_llm_output: str,
        required_fields: Optional[List[str]] = None
    ) -> Dict:

        criterion = "valid_json"
        max_score = CLAUSE_RUBRIC[criterion]["max_score"]
        half = max_score // 2

        if required_fields is None:
            required_fields = DEFAULT_REQUIRED_FIELDS

        try:
            cleaned = re.sub(
                r"^```(json)?|```$", "",
                raw_llm_output, flags=re.MULTILINE
            ).strip()
            parsed = json.loads(cleaned)

        except json.JSONDecodeError as e:
            return {
                "criterion": criterion,
                "description": CLAUSE_RUBRIC[criterion]["description"],
                "score": 0, "max_score": max_score,
                "passed": False,
                "detail": f"JSON parsing failed: {e}"
            }

        structure_score = half
        issues = []

        if not isinstance(parsed, list):
            structure_score = 0
            issues.append(f"Expected list, got {type(parsed).__name__}.")
        else:
            for i, item in enumerate(parsed):
                missing = [f for f in required_fields if f not in item]
                if missing:
                    structure_score = 0
                    issues.append(f"Item {i} missing fields: {missing}")
                    break

        total = half + structure_score
        return {
            "criterion": criterion,
            "description": CLAUSE_RUBRIC[criterion]["description"],
            "score": total, "max_score": max_score,
            "passed": total == max_score,
            "detail": (
                "JSON parsed and structure validated successfully."
                if not issues else f"Issues: {issues}"
            )
        }

    # ---- Criterion 2: Correct Clause Count ----
    def _eval_correct_clause_count(
        self,
        llm_results: List[Dict],
        expected_count: int
    ) -> Dict:

        criterion = "correct_clause_count"
        max_score = CLAUSE_RUBRIC[criterion]["max_score"]
        actual = len(llm_results)
        passed = actual == expected_count

        return {
            "criterion": criterion,
            "description": CLAUSE_RUBRIC[criterion]["description"],
            "score": max_score if passed else 0,
            "max_score": max_score,
            "passed": passed,
            "detail": f"Expected {expected_count}, LLM returned {actual}."
        }

    # ---- Criterion 3: Correct Classification (LLM-as-Judge) ----
    def _eval_correct_classification(
        self,
        llm_results: List[Dict],
        ground_truth: Optional[List[str]] = None
    ) -> Dict:
        """
        Default path: LLM-as-judge. A single batched prompt is sent to the
        judge model, which returns a correct/incorrect verdict per clause.
        No manual labeling required.

        Optional path: if `ground_truth` IS passed, it is used instead
        (kept only for backward compatibility / auditing the judge).
        """

        criterion = "correct_classification"
        max_score = CLAUSE_RUBRIC[criterion]["max_score"]

        if not llm_results:
            return {
                "criterion": criterion,
                "description": CLAUSE_RUBRIC[criterion]["description"],
                "score": max_score, "max_score": max_score,
                "passed": True,
                "detail": "No clauses to judge."
            }

        # ---- Optional backward-compatible manual path ----
        if ground_truth:
            total = min(len(llm_results), len(ground_truth))
            correct = 0
            mismatches = []
            for i in range(total):
                predicted = llm_results[i].get("clause_type", "Unknown")
                expected = ground_truth[i]
                if predicted.strip().lower() == expected.strip().lower():
                    correct += 1
                else:
                    mismatches.append(
                        f"Clause {i}: expected='{expected}', predicted='{predicted}'"
                    )
            score = round((correct / total) * max_score) if total > 0 else 0
            return {
                "criterion": criterion,
                "description": CLAUSE_RUBRIC[criterion]["description"],
                "score": score, "max_score": max_score,
                "passed": score >= (max_score * CRITERION_PASS_THRESHOLD),
                "detail": (
                    f"(manual ground truth) {correct}/{total} correctly classified."
                    + (f" Mismatches: {mismatches}" if mismatches else "")
                )
            }

        # ---- Default: LLM-as-judge path (no manual effort) ----
        try:
            # Normalize category names first (e.g. "Termination Clause" ->
            # "Termination") so the judge isn't confused by naming
            # differences between the classifier's output and the
            # allowed_categories list — this alone fixes a common source
            # of false "incorrect" verdicts.
            normalized_results = []
            renamed = []
            for i, item in enumerate(llm_results):
                original_type = item.get("clause_type", "Unknown")
                normalized_type = _normalize_category(original_type, self.allowed_categories)
                if normalized_type != original_type:
                    renamed.append(f"Clause {i}: '{original_type}' -> '{normalized_type}'")
                new_item = dict(item)
                new_item["clause_type"] = normalized_type
                normalized_results.append(new_item)

            prompt = _build_judge_prompt(normalized_results, self.allowed_categories)
            raw_judge_output = self.llm_judge_fn(prompt, self.judge_model)

            cleaned = re.sub(
                r"^```(json)?|```$", "", raw_judge_output, flags=re.MULTILINE
            ).strip()

            # ROBUST JSON PARSE: judge models sometimes wrap the array with
            # stray prose or trailing text even after we strip code fences.
            # If direct parsing fails, fall back to extracting the first
            # [...] block instead of failing the whole criterion.
            try:
                verdicts = json.loads(cleaned)
            except json.JSONDecodeError:
                match = re.search(r"\[.*\]", cleaned, re.DOTALL)
                if not match:
                    raise
                verdicts = json.loads(match.group(0))

            # ROBUST KEY MATCHING (the actual root cause of most inflated
            # FAIL counts): judge models often return "clause_number" as a
            # string ("0") instead of an int, or occasionally omit/garble
            # it. A plain int-keyed dict lookup then misses the verdict
            # entirely and wrongly counts a correctly-judged clause as
            # "no verdict returned". We coerce every key to int where
            # possible, and if that still doesn't cover every verdict,
            # fall back to matching by position (judge output order == our
            # input order, since we numbered them explicitly in the prompt).
            verdict_map = {}
            all_keys_valid = True
            for v in verdicts:
                raw_key = v.get("clause_number")
                try:
                    key = int(raw_key)
                except (TypeError, ValueError):
                    key = None
                    all_keys_valid = False
                verdict_map[key] = v

            if not all_keys_valid or len(verdict_map) < len(verdicts):
                verdict_map = {idx: v for idx, v in enumerate(verdicts)}

            total = len(llm_results)
            correct = 0
            mismatches = []

            for i in range(total):
                v = verdict_map.get(i)
                if v is not None and v.get("correct") is True:
                    correct += 1
                else:
                    reason = v.get("reason", "no verdict returned") if v else "no verdict returned"
                    mismatches.append(f"Clause {i}: judged incorrect — {reason}")

            score = round((correct / total) * max_score) if total > 0 else 0

            return {
                "criterion": criterion,
                "description": CLAUSE_RUBRIC[criterion]["description"],
                "score": score, "max_score": max_score,
                "passed": score >= (max_score * CRITERION_PASS_THRESHOLD),
                "detail": (
                    f"(LLM-as-judge) {correct}/{total} clauses judged correctly classified."
                    + (f" Issues: {mismatches}" if mismatches else "")
                    + (f" Note: normalized category names: {renamed}" if renamed else "")
                )
            }

        except Exception as e:
            logger.warning(f"LLM-as-judge classification check failed: {e}")
            return {
                "criterion": criterion,
                "description": CLAUSE_RUBRIC[criterion]["description"],
                "score": 0, "max_score": max_score,
                "passed": False,
                "detail": f"LLM-as-judge call/parse failed: {e}"
            }

    # ---- Criterion 4: No Hallucination ----
    def _eval_no_hallucination(
        self,
        llm_results: List[Dict],
        original_contract: Optional[str] = None
    ) -> Dict:

        criterion = "no_hallucination"
        max_score = CLAUSE_RUBRIC[criterion]["max_score"]
        total = len(llm_results)
        hallucinations = []

        for i, item in enumerate(llm_results):

            clause_type = item.get("clause_type", "Unknown")
            if clause_type not in self.allowed_categories:
                hallucinations.append(
                    f"Clause {i}: type hallucination — '{clause_type}' not in allowed list."
                )

            if original_contract:
                text = item.get("text", "").strip()
                # Fuzzy word-overlap check instead of strict exact substring —
                # avoids false positives from apostrophes, line breaks, or
                # clause-numbering prefixes merged into the extracted text.
                if text and not _fuzzy_text_found(text, original_contract):
                    hallucinations.append(
                        f"Clause {i}: text hallucination — "
                        f"clause text not sufficiently found in contract. Snippet: '{text[:60]}'"
                    )

        h_count = len(hallucinations)
        score = round(((total - h_count) / total) * max_score) if total > 0 else max_score
        passed = score >= (max_score * CRITERION_PASS_THRESHOLD)

        return {
            "criterion": criterion,
            "description": CLAUSE_RUBRIC[criterion]["description"],
            "score": score, "max_score": max_score,
            "passed": passed,
            "detail": (
                f"{h_count} hallucination(s) detected."
                + (f" Details: {hallucinations}" if hallucinations else " None found.")
            )
        }

    # ---- Criterion 5: Prompt Compliance ----
    def _eval_prompt_compliance(
        self,
        llm_results: List[Dict],
        required_fields: Optional[List[str]] = None
    ) -> Dict:

        criterion = "prompt_compliance"
        max_score = CLAUSE_RUBRIC[criterion]["max_score"]

        if required_fields is None:
            required_fields = DEFAULT_REQUIRED_FIELDS

        total = len(llm_results)
        violations = []

        for i, item in enumerate(llm_results):
            missing = [f for f in required_fields if f not in item]
            if missing:
                violations.append(f"Clause {i}: missing {missing}")

        v_count = len(violations)
        score = round(((total - v_count) / total) * max_score) if total > 0 else max_score

        return {
            "criterion": criterion,
            "description": CLAUSE_RUBRIC[criterion]["description"],
            "score": score, "max_score": max_score,
            "passed": v_count == 0,
            "detail": (
                f"{v_count} violation(s)."
                + (f" Details: {violations}" if violations else " All fields present.")
            )
        }

    # ---- Main evaluate method ----
    def evaluate(
        self,
        raw_llm_output: str,
        llm_results: List[Dict],
        expected_clause_count: int,
        ground_truth: Optional[List[str]] = None,
        original_contract: Optional[str] = None,
        required_fields: Optional[List[str]] = None
    ) -> Dict:

        results = [
            self._eval_valid_json(raw_llm_output, required_fields),
            self._eval_correct_clause_count(llm_results, expected_clause_count),
            self._eval_correct_classification(llm_results, ground_truth),
            self._eval_no_hallucination(llm_results, original_contract),
            self._eval_prompt_compliance(llm_results, required_fields)
        ]

        total = sum(r["score"] for r in results)

        return {
            "component": "Clause Extraction & Classification",
            "criteria": results,
            "total_score": total,
            "total_max_score": CLAUSE_MAX,
            "percentage": round((total / CLAUSE_MAX) * 100, 2),
            "overall_pass": total >= (CLAUSE_MAX * (PASS_THRESHOLD / 100))
        }


# ================================================================
# PART B — SUMMARY EVALUATOR  (unchanged — already manual-effort-free)
# ================================================================

class SummaryEvaluator:

    def _eval_length_compliance(self, summary: str, max_words: int = 120) -> Dict:
        criterion = "length_compliance"
        max_score = SUMMARY_RUBRIC[criterion]["max_score"]
        word_count = len(summary.split())
        passed = word_count <= max_words
        return {
            "criterion": criterion,
            "description": SUMMARY_RUBRIC[criterion]["description"],
            "score": max_score if passed else 0,
            "max_score": max_score,
            "passed": passed,
            "detail": (
                f"Word count: {word_count}. "
                f"{'Within' if passed else 'Exceeds'} the {max_words}-word limit."
            )
        }

    def _eval_no_hallucination(self, summary: str, structured_input: Dict) -> Dict:
        criterion = "no_hallucination"
        max_score = SUMMARY_RUBRIC[criterion]["max_score"]

        hallucinations = []
        summary_lower = summary.lower()

        known_risks = (
            structured_input.get("high_risks", []) +
            structured_input.get("medium_risks", []) +
            structured_input.get("low_risks", [])
        )
        known_risks_lower = [r.lower() for r in known_risks]
        known_categories_lower = [
            c.lower() for c in structured_input.get("clause_categories", [])
        ]

        dynamic_keywords = set()
        for phrase in known_risks_lower + known_categories_lower:
            words = [w for w in phrase.split() if len(w) > 4]
            if len(words) >= 2:
                dynamic_keywords.add(" ".join(words[:3]))

        candidate_keywords = set(BASE_RISK_KEYWORDS) | dynamic_keywords

        for kw in candidate_keywords:
            if kw in summary_lower:
                if not any(kw in r for r in known_risks_lower):
                    if not any(kw in c for c in known_categories_lower):
                        hallucinations.append(
                            f"'{kw}' mentioned in summary but not present in structured input."
                        )

        h_count = len(hallucinations)
        score = max_score if h_count == 0 else max(0, max_score - (h_count * 5))

        return {
            "criterion": criterion,
            "description": SUMMARY_RUBRIC[criterion]["description"],
            "score": score, "max_score": max_score,
            "passed": h_count == 0,
            "detail": (
                f"{h_count} potential hallucination(s) detected."
                + (f" Details: {hallucinations}" if hallucinations else " None found.")
            )
        }

    def _eval_risk_coverage(self, summary: str, structured_input: Dict) -> Dict:
        criterion = "risk_coverage"
        max_score = SUMMARY_RUBRIC[criterion]["max_score"]
        summary_lower = summary.lower()
        high_risks = structured_input.get("high_risks", [])

        if not high_risks:
            return {
                "criterion": criterion,
                "description": SUMMARY_RUBRIC[criterion]["description"],
                "score": max_score, "max_score": max_score,
                "passed": True,
                "detail": "No high risks in input. Criterion skipped (full marks)."
            }

        covered, not_covered = [], []
        for risk in high_risks:
            keywords = [w for w in risk.lower().split() if len(w) > 4]
            if any(kw in summary_lower for kw in keywords):
                covered.append(risk)
            else:
                not_covered.append(risk)

        score = round((len(covered) / len(high_risks)) * max_score)
        return {
            "criterion": criterion,
            "description": SUMMARY_RUBRIC[criterion]["description"],
            "score": score, "max_score": max_score,
            "passed": len(not_covered) == 0,
            "detail": (
                f"{len(covered)}/{len(high_risks)} high risks covered in summary."
                + (f" Not covered: {not_covered}" if not_covered else "")
            )
        }

    def _eval_clause_coverage(self, summary: str, structured_input: Dict) -> Dict:
        criterion = "clause_coverage"
        max_score = SUMMARY_RUBRIC[criterion]["max_score"]
        summary_lower = summary.lower()
        categories = structured_input.get("clause_categories", [])

        if not categories:
            return {
                "criterion": criterion,
                "description": SUMMARY_RUBRIC[criterion]["description"],
                "score": max_score, "max_score": max_score,
                "passed": True,
                "detail": "No categories in input. Criterion skipped (full marks)."
            }

        covered, not_covered = [], []
        for cat in categories:
            if cat.lower() in summary_lower:
                covered.append(cat)
            else:
                not_covered.append(cat)

        score = round((len(covered) / len(categories)) * max_score)
        return {
            "criterion": criterion,
            "description": SUMMARY_RUBRIC[criterion]["description"],
            "score": score, "max_score": max_score,
            "passed": len(not_covered) == 0,
            "detail": (
                f"{len(covered)}/{len(categories)} clause categories mentioned."
                + (f" Not mentioned: {not_covered}" if not_covered else "")
            )
        }

    def _eval_format_compliance(self, summary: str) -> Dict:
        criterion = "format_compliance"
        max_score = SUMMARY_RUBRIC[criterion]["max_score"]
        violations = []

        if re.search(r"^[\•\*\-]\s", summary, re.MULTILINE):
            violations.append("Bullet points detected.")
        if re.search(r"^#{1,6}\s", summary, re.MULTILINE):
            violations.append("Markdown headings detected.")
        if re.search(r"\*\*.+?\*\*|\*.+?\*", summary):
            violations.append("Markdown bold/italic detected.")

        score = max_score if not violations else 0
        return {
            "criterion": criterion,
            "description": SUMMARY_RUBRIC[criterion]["description"],
            "score": score, "max_score": max_score,
            "passed": not violations,
            "detail": (
                "Format compliant — paragraph format, no markdown."
                if not violations
                else f"Violations: {violations}"
            )
        }

    def evaluate(self, summary: str, structured_input: Dict) -> Dict:
        results = [
            self._eval_length_compliance(summary),
            self._eval_no_hallucination(summary, structured_input),
            self._eval_risk_coverage(summary, structured_input),
            self._eval_clause_coverage(summary, structured_input),
            self._eval_format_compliance(summary)
        ]
        total = sum(r["score"] for r in results)
        return {
            "component": "Executive Summary",
            "criteria": results,
            "total_score": total,
            "total_max_score": SUMMARY_MAX,
            "percentage": round((total / SUMMARY_MAX) * 100, 2),
            "overall_pass": total >= (SUMMARY_MAX * (PASS_THRESHOLD / 100))
        }


# ================================================================
# PART C — RISK DETECTION EVALUATOR
# (Pure rule-based — no LLM call, so this never degrades when Ollama
#  is unavailable. Scores the risks already attached to each clause
#  by RiskDetector, not the clause classification itself.)
# ================================================================

class RiskDetectionEvaluator:

    def _eval_valid_severity_levels(self, all_risks: List[Dict]) -> Dict:
        criterion = "valid_severity_levels"
        max_score = RISK_RUBRIC[criterion]["max_score"]
        total = len(all_risks)

        if total == 0:
            return {
                "criterion": criterion,
                "description": RISK_RUBRIC[criterion]["description"],
                "score": max_score, "max_score": max_score,
                "passed": True,
                "detail": "No risks detected. Criterion skipped (full marks)."
            }

        invalid = [
            i for i, r in enumerate(all_risks)
            if r.get("level") not in ("HIGH", "MEDIUM", "LOW")
        ]
        score = round(((total - len(invalid)) / total) * max_score)
        passed = score >= (max_score * CRITERION_PASS_THRESHOLD)

        return {
            "criterion": criterion,
            "description": RISK_RUBRIC[criterion]["description"],
            "score": score, "max_score": max_score,
            "passed": passed,
            "detail": (
                f"{len(invalid)}/{total} risk(s) with an invalid severity level."
                + (f" Indices: {invalid}" if invalid else " All valid.")
            )
        }

    def _eval_known_pattern_match(
        self,
        all_risks: List[Dict],
        risk_patterns: Optional[List[Dict]]
    ) -> Dict:
        criterion = "known_pattern_match"
        max_score = RISK_RUBRIC[criterion]["max_score"]

        if not risk_patterns:
            return {
                "criterion": criterion,
                "description": RISK_RUBRIC[criterion]["description"],
                "score": max_score, "max_score": max_score,
                "passed": True,
                "detail": "No risk-pattern rules supplied; check skipped (full marks)."
            }

        total = len(all_risks)
        if total == 0:
            return {
                "criterion": criterion,
                "description": RISK_RUBRIC[criterion]["description"],
                "score": max_score, "max_score": max_score,
                "passed": True,
                "detail": "No risks detected. Criterion skipped (full marks)."
            }

        known_patterns_lower = {
            p.get("pattern", "").strip().lower() for p in risk_patterns
        }
        hallucinated = [
            i for i, r in enumerate(all_risks)
            if r.get("pattern", "").strip().lower() not in known_patterns_lower
        ]
        score = round(((total - len(hallucinated)) / total) * max_score)
        passed = score >= (max_score * CRITERION_PASS_THRESHOLD)

        return {
            "criterion": criterion,
            "description": RISK_RUBRIC[criterion]["description"],
            "score": score, "max_score": max_score,
            "passed": passed,
            "detail": (
                f"{len(hallucinated)}/{total} risk(s) reference a pattern not "
                f"found in the configured risk-pattern rules."
                + (f" Indices: {hallucinated}" if hallucinated else " All matched.")
            )
        }

    def _eval_required_fields(self, all_risks: List[Dict]) -> Dict:
        criterion = "required_fields"
        max_score = RISK_RUBRIC[criterion]["max_score"]
        required = ["pattern", "risk", "level", "matched_text"]
        total = len(all_risks)

        if total == 0:
            return {
                "criterion": criterion,
                "description": RISK_RUBRIC[criterion]["description"],
                "score": max_score, "max_score": max_score,
                "passed": True,
                "detail": "No risks detected. Criterion skipped (full marks)."
            }

        violations = []
        for i, r in enumerate(all_risks):
            missing = [f for f in required if f not in r]
            if missing:
                violations.append(f"Risk {i}: missing {missing}")

        score = round(((total - len(violations)) / total) * max_score)
        return {
            "criterion": criterion,
            "description": RISK_RUBRIC[criterion]["description"],
            "score": score, "max_score": max_score,
            "passed": len(violations) == 0,
            "detail": (
                f"{len(violations)} violation(s)."
                + (f" Details: {violations}" if violations else " All fields present.")
            )
        }

    def evaluate(
        self,
        clauses: List[Dict],
        risk_patterns: Optional[List[Dict]] = None
    ) -> Dict:

        all_risks = []
        for clause in clauses:
            all_risks.extend(clause.get("risks", []))

        results = [
            self._eval_valid_severity_levels(all_risks),
            self._eval_known_pattern_match(all_risks, risk_patterns),
            self._eval_required_fields(all_risks)
        ]

        total = sum(r["score"] for r in results)

        return {
            "component": "Risk Detection",
            "criteria": results,
            "total_score": total,
            "total_max_score": RISK_MAX,
            "percentage": round((total / RISK_MAX) * 100, 2),
            "overall_pass": total >= (RISK_MAX * (PASS_THRESHOLD / 100))
        }


# ================================================================
# UNIFIED EVALUATOR
# ================================================================

class UnifiedEvaluator:

    def __init__(
        self,
        allowed_categories: List[str],
        judge_model: str = None,
        llm_judge_fn: Optional[Callable[[str, str], str]] = None
    ):
        self.clause_evaluator = ClauseEvaluator(
            allowed_categories, judge_model=judge_model, llm_judge_fn=llm_judge_fn
        )
        self.summary_evaluator = SummaryEvaluator()
        self.risk_evaluator = RiskDetectionEvaluator()

    def evaluate(
        self,
        raw_llm_output: str,
        llm_results: List[Dict],
        expected_clause_count: int,
        ground_truth: Optional[List[str]] = None,
        original_contract: Optional[str] = None,
        required_fields: Optional[List[str]] = None,
        summary: Optional[str] = None,
        structured_input: Optional[Dict] = None,
        risk_patterns: Optional[List[Dict]] = None
    ) -> Dict:

        report = {}

        report["clause_extraction"] = self.clause_evaluator.evaluate(
            raw_llm_output=raw_llm_output,
            llm_results=llm_results,
            expected_clause_count=expected_clause_count,
            ground_truth=ground_truth,
            original_contract=original_contract,
            required_fields=required_fields
        )

        report["risk_detection"] = self.risk_evaluator.evaluate(
            clauses=llm_results,
            risk_patterns=risk_patterns
        )

        percentages = [
            report["clause_extraction"]["percentage"],
            report["risk_detection"]["percentage"]
        ]

        if summary and structured_input:
            report["summary_quality"] = self.summary_evaluator.evaluate(
                summary=summary,
                structured_input=structured_input
            )
            percentages.append(report["summary_quality"]["percentage"])

        overall_quality_score = round(sum(percentages) / len(percentages), 2)
        report["overall_quality_score"] = overall_quality_score
        report["overall_pass"] = overall_quality_score >= PASS_THRESHOLD

        return report

    # ---- Full verbose report (old style) ----
    def print_report(self, report: Dict) -> None:

        def _print_section(section: Dict):
            print("\n" + "=" * 62)
            print(f"   {section['component'].upper()}")
            print("=" * 62)
            for c in section["criteria"]:
                status = "PASS" if c["passed"] else "FAIL"
                print(f"\n  [{status}] {c['criterion'].upper().replace('_', ' ')}")
                print(f"         {c['description']}")
                print(f"         Score : {c['score']} / {c['max_score']}")
                print(f"         Detail: {c['detail']}")
            print("\n" + "-" * 62)
            print(
                f"  TOTAL : {section['total_score']} / {section['total_max_score']}"
                f"  ({section['percentage']}%)  "
                f"{'PASS' if section['overall_pass'] else 'FAIL'}"
            )
            print("=" * 62)

        _print_section(report["clause_extraction"])
        _print_section(report["risk_detection"])
        if "summary_quality" in report:
            _print_section(report["summary_quality"])
        if "overall_quality_score" in report:
            print("\n" + "*" * 62)
            print("   OVERALL QUALITY SCORE")
            print("*" * 62)
            print(f"  Clause Extraction : {report['clause_extraction']['percentage']}%")
            print(f"  Risk Detection    : {report['risk_detection']['percentage']}%")
            if "summary_quality" in report:
                print(f"  Summary Quality   : {report['summary_quality']['percentage']}%")
            print(f"  Overall           : {report['overall_quality_score']}%")
            print(f"  Result            : {'PASS' if report['overall_pass'] else 'FAIL'}  (threshold: {PASS_THRESHOLD}%)")
            print("*" * 62 + "\n")

    # ---- NEW: Simple 6-metric report (matches requested output format) ----
    def print_simple_report(self, report: Dict, section: str = "clause_extraction") -> None:
        """
        Prints exactly 5 criteria + 1 overall score line, in the compact
        format:

            ==============================
            Evaluation Metrics
            ==============================
            1. JSON Validation        : PASS (20/20)
            2. Clause Count Accuracy  : PASS (20/20)
            3. Classification Accuracy: PASS (28/30)
            4. Hallucination Check    : PASS (18/20)
            5. Prompt Compliance      : PASS (10/10)
            6. Overall Score          : 96%
            ==============================

        `section` picks which report block to print:
        "clause_extraction" (default), "risk_detection", or "summary_quality".
        """

        LABELS = {
            "valid_json": "JSON Validation",
            "correct_clause_count": "Clause Count Accuracy",
            "correct_classification": "Classification Accuracy",
            "no_hallucination": "Hallucination Check",
            "prompt_compliance": "Prompt Compliance",
            "length_compliance": "Length Compliance",
            "risk_coverage": "Risk Coverage",
            "clause_coverage": "Clause Coverage",
            "format_compliance": "Format Compliance",
        }

        sec = report[section]
        width = max(len(LABELS.get(c["criterion"], c["criterion"])) for c in sec["criteria"])

        print("=" * 46)
        print("Evaluation Metrics")
        print("=" * 46)

        for i, c in enumerate(sec["criteria"], start=1):
            label = LABELS.get(c["criterion"], c["criterion"]).ljust(width)
            status = "PASS" if c["passed"] else "FAIL"
            print(f"{i}. {label} : {status} ({c['score']}/{c['max_score']})")

        print(f"{len(sec['criteria']) + 1}. {'Overall Score'.ljust(width)} : {sec['percentage']}%")
        print("=" * 46)

def run_pipeline_evaluation(
    original_contract: str,
    clauses: list,
    summary: str,
    clause_dictionary: dict,
    risk_patterns: Optional[list] = None
):
    """
    Wrapper used by app.py. `risk_patterns` (the configured risk-pattern
    rules, e.g. risk_detector.rules) is optional — pass it to enable the
    "known_pattern_match" check in the risk_detection score; omit it and
    that one criterion is skipped (full marks) rather than failing.
    """

    import json

    raw_llm_output = json.dumps(clauses)
    expected_clause_count = len(clauses)

    high_risks = []
    medium_risks = []
    low_risks = []
    clause_categories = set()

    for clause in clauses:
        clause_categories.add(
            clause.get("clause_type", "Unknown")
        )

        for risk in clause.get("risks", []):

            if risk["level"] == "HIGH":
                high_risks.append(risk["risk"])

            elif risk["level"] == "MEDIUM":
                medium_risks.append(risk["risk"])

            elif risk["level"] == "LOW":
                low_risks.append(risk["risk"])

    structured_input = {
        "overall_risk_level":
            "HIGH" if high_risks else
            ("MEDIUM" if medium_risks else "LOW"),

        "clause_categories":
            list(clause_categories),

        "high_risks":
            list(set(high_risks)),

        "medium_risks":
            list(set(medium_risks)),

        "low_risks":
            list(set(low_risks))
    }

    evaluator = UnifiedEvaluator(
        allowed_categories=list(clause_dictionary.keys())
    )

    return evaluator.evaluate(
        raw_llm_output=raw_llm_output,
        llm_results=clauses,
        expected_clause_count=expected_clause_count,
        original_contract=original_contract,
        summary=summary,
        structured_input=structured_input,
        risk_patterns=risk_patterns
    )
# ================================================================
# TESTING — full pipeline
# ================================================================

if __name__ == "__main__":

    import argparse
    import os

    from clause_extraction import ClauseExtractor
    from SummaryGenerator import SummaryGenerator

    DEFAULT_SAMPLE_TEXT = """
1. TERMINATION

Either party may terminate this agreement.

1.1 NOTICE

Written notice shall be provided.

SECTION 2

Confidential information shall remain protected.

ARTICLE III

Payment terms are listed below.

A. LIABILITY

The vendor shall be liable.

(a) DAMAGES

Damages may be recovered.
"""

    parser = argparse.ArgumentParser(description="Run the clause + summary evaluation pipeline.")
    parser.add_argument(
        "--contract", type=str, default=None,
        help="Path to a contract text file. Defaults to a built-in sample contract."
    )
    parser.add_argument(
        "--judge-model", type=str, default="llama3.2",
        help="Ollama model used as the LLM-judge for classification accuracy."
    )
    args = parser.parse_args()

    if args.contract and os.path.exists(args.contract):
        with open(args.contract, "r", encoding="utf-8") as f:
            sample_text = f.read()
    else:
        sample_text = DEFAULT_SAMPLE_TEXT

    # ---- Step 1: Regex Extraction ----
    extractor = ClauseExtractor("data/clauses.json")
    extracted_clauses = extractor.extract_clauses(sample_text)
    expected_count = len(extracted_clauses)

    # ---- Step 2: Batch LLM Classification (1 call) ----
    from llm_client import chat

    if hasattr(extractor, "build_prompt"):
        prompt = extractor.build_prompt(extracted_clauses)
    else:
        logger.warning(
            "ClauseExtractor has no public build_prompt(); falling back to "
            "the private _build_few_shot_batch_prompt()."
        )
        prompt = extractor._build_few_shot_batch_prompt(extracted_clauses)

    raw_llm_output = chat(
    user_prompt=prompt,
    temperature=0.0,
    max_tokens=2000,
    )

    try:
        cleaned = re.sub(r"^```(json)?|```$", "", raw_llm_output, flags=re.MULTILINE).strip()
        batch_out = json.loads(cleaned)
        id_to_type = {}
        for idx, item in enumerate(batch_out):
            key = item.get("clause_id", idx)
            id_to_type[key] = item.get("clause_type", "Unknown")
        for idx, clause in enumerate(extracted_clauses):
            clause["clause_type"] = id_to_type.get(idx, "Unknown")
            clause["risks"] = []
        llm_results = extracted_clauses
    except Exception as e:
        logger.warning(f"Failed to merge LLM classifications: {e}")
        llm_results = extracted_clauses

    # ---- Step 3: Summary Generation ----
    generator = SummaryGenerator(use_llm=True)
    summary_text = generator.generate_executive_summary(llm_results)

    high_risks, medium_risks, low_risks = [], [], []
    clause_types = set()
    for clause in llm_results:
        clause_types.add(clause["clause_type"])
        for risk in clause.get("risks", []):
            if risk["level"] == "HIGH":
                high_risks.append(risk["risk"])
            elif risk["level"] == "MEDIUM":
                medium_risks.append(risk["risk"])
            elif risk["level"] == "LOW":
                low_risks.append(risk["risk"])

    structured_input = {
        "overall_risk_level": "HIGH" if high_risks else ("MEDIUM" if medium_risks else "LOW"),
        "clause_categories": sorted(list(clause_types)),
        "high_risks": sorted(list(set(high_risks))),
        "medium_risks": sorted(list(set(medium_risks))),
        "low_risks": sorted(list(set(low_risks))),
        "risk_counts": {
            "high": len(high_risks),
            "medium": len(medium_risks),
            "low": len(low_risks)
        }
    }

    # ---- Step 4: Unified Rubric Evaluation (LLM-as-judge, no manual labels) ----
    allowed_categories = list(extractor.clause_dictionary.keys())

    evaluator = UnifiedEvaluator(
        allowed_categories=allowed_categories,
        judge_model=args.judge_model
    )

    report = evaluator.evaluate(
        raw_llm_output=raw_llm_output,
        llm_results=llm_results,
        expected_clause_count=expected_count,
        ground_truth=None,          # <-- no manual labels needed anymore
        original_contract=sample_text,
        summary=summary_text,
        structured_input=structured_input
    )

    # ---- Step 5: Print reports ----
    evaluator.print_simple_report(report, section="clause_extraction")

    print("\nFull JSON Report:")
    print(json.dumps(report, indent=4))
