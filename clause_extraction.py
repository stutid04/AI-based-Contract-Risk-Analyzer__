"""
clause_extraction.py

Clause Extraction and Classification Module
(Hybrid architecture: Regex extraction -> Per-clause LLM classification
 -> Keyword-based fallback classifier, using local Llama 3.2 via Ollama)
"""

import json
import logging
import re
from llm_client import chat
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClauseExtractor:
    """
    Extracts and classifies clauses from legal contracts.
    """

    def __init__(
        self,
        clause_json_path: str,
        threshold: int = 1,
        ollama_model: str = "llama3.2"
    ):
        """
        Load clause categories from JSON.
        ollama_model -> local LLM model name (must be `ollama pull`-ed already).
        """

        self.threshold = threshold
        self.ollama_model = ollama_model

        try:
            with open(
                clause_json_path,
                "r",
                encoding="utf-8"
            ) as file:

                self.clause_dictionary = json.load(file)

        except Exception as e:

            logger.error(
                f"Unable to load JSON file: {e}"
            )

            raise

    # ----------------------------------------------------------------
    # STEP 1: REGEX-BASED EXTRACTION (unchanged from your original code)
    # ----------------------------------------------------------------
    def extract_clauses(
        self,
        text: str
    ) -> List[Dict]:
        """
        Extract clauses using common legal numbering patterns.
        """

        try:

            clause_pattern = re.compile(
                r"""
                (?=^(
                    \d+(?:\.\d+)*\.?\s+.*      |
                    section\s+\d+.*            |
                    article\s+[ivxlcdm]+.*     |
                    [A-Z]\.\s+.*               |
                    \([a-z]\)\s+.*
                ))
                """,
                re.IGNORECASE
                | re.MULTILINE
                | re.VERBOSE
            )

            matches = list(
                clause_pattern.finditer(text)
            )

            clauses = []

            if not matches:
                return clauses

            for i in range(len(matches)):

                start = matches[i].start()

                if i + 1 < len(matches):
                    end = matches[i + 1].start()
                else:
                    end = len(text)

                block = text[start:end].strip()

                lines = block.splitlines()

                if not lines:
                    continue

                heading = lines[0].strip()

                clause_number = ""
                clause_title = heading

                number_match = re.match(
                    r'^(\d+(?:\.\d+)*\.?)',
                    heading
                )

                section_match = re.match(
                    r'^(section\s+\d+)',
                    heading,
                    re.IGNORECASE
                )

                article_match = re.match(
                    r'^(article\s+[ivxlcdm]+)',
                    heading,
                    re.IGNORECASE
                )

                alpha_match = re.match(
                    r'^([A-Z]\.)',
                    heading
                )

                bracket_match = re.match(
                    r'^(\([a-z]\))',
                    heading
                )

                if number_match:

                    clause_number = (
                        number_match.group(1)
                    )

                    clause_title = heading[
                        len(clause_number):
                    ].strip()

                elif section_match:

                    clause_number = (
                        section_match.group(1)
                    )

                    clause_title = heading

                elif article_match:

                    clause_number = (
                        article_match.group(1)
                    )

                    clause_title = heading

                elif alpha_match:

                    clause_number = (
                        alpha_match.group(1)
                    )

                    clause_title = heading[
                        len(clause_number):
                    ].strip()

                elif bracket_match:

                    clause_number = (
                        bracket_match.group(1)
                    )

                    clause_title = heading[
                        len(clause_number):
                    ].strip()

                clauses.append(
                    {
                        "clause_number":
                            clause_number,
                        "clause_title":
                            clause_title,
                        "text":
                            block
                    }
                )

            return clauses

        except Exception as e:

            logger.error(
                f"Clause extraction failed: {e}"
            )

            return []

    # ----------------------------------------------------------------
    # STEP 2 (fallback): KEYWORD-BASED CLASSIFICATION
    # (unchanged from your original code)
    # ----------------------------------------------------------------
    def classify_clause(
        self,
        clause_text: str
    ) -> str:
        """
        Classify clause using keyword scoring.
        Used as a fallback when the LLM is unavailable or fails.
        """

        try:

            clause_text_lower = clause_text.lower()

            scores = {}

            for (
                category,
                keywords
            ) in self.clause_dictionary.items():

                score = 0

                for keyword in keywords:

                    score += clause_text_lower.count(
                        keyword.lower()
                    )

                scores[category] = score

            best_category = max(
                scores,
                key=scores.get,
                default="Unknown"
            )

            if (
                scores[best_category]
                < self.threshold
            ):
                return "Unknown"

            return best_category

        except Exception as e:

            logger.error(
                f"Keyword classification failed: {e}"
            )

            return "Unknown"

    # ----------------------------------------------------------------
    # STEP 2 (primary): LLM-BASED CLASSIFICATION (per clause)
    # ----------------------------------------------------------------
    def _build_few_shot_classification_prompt(
        self,
        clause_text: str
    ) -> str:
        """
        Builds a strict few-shot prompt to classify a SINGLE clause
        (not the whole contract). This is faster, more accurate,
        and far less prone to hallucination than feeding the whole
        contract to the LLM at once.
        """

        categories = list(self.clause_dictionary.keys())

        system_rules = f"""You are a strict legal contract clause classifier.

IMPORTANT RULES:
- Use ONLY the given clause text. Do NOT use external knowledge.
- Extract ONLY the clause type explicitly present in the text. Do NOT invent or assume anything.
- Choose exactly ONE category from this list: {categories}
- If the clause does not clearly match any category in the list, return "Unknown".
- Return ONLY valid JSON in the exact format shown in the examples. No explanations, no markdown.
"""

        example_1_input = "Either party may terminate this agreement by giving 30 days written notice."
        example_1_output = json.dumps({"clause_type": "Termination"})

        example_2_input = "The client shall pay the vendor within 15 days of invoice receipt."
        example_2_output = json.dumps({"clause_type": "Payment"})

        example_3_input = "This document is for informational purposes only and creates no obligations."
        example_3_output = json.dumps({"clause_type": "Unknown"})

        prompt = f"""{system_rules}

Example 1:
Clause: "{example_1_input}"
Output: {example_1_output}

Example 2:
Clause: "{example_2_input}"
Output: {example_2_output}

Example 3:
Clause: "{example_3_input}"
Output: {example_3_output}

Now classify the following clause.

Clause: "{clause_text}"
Output:
"""

        return prompt

    def classify_clause_llm(
        self,
        clause_text: str,
        temperature: float = 0.0
    ) -> str:
        """
        Classify a single clause using local Llama 3.2 via the
        official `ollama` Python package.

        Requires Ollama running locally:
            ollama pull llama3.2

        Falls back to keyword-based classification if the LLM call
        fails or returns invalid output.
        """

        prompt = self._build_few_shot_classification_prompt(clause_text)

        try:

            raw_output = chat(
               user_prompt=prompt,
               temperature=temperature,
               max_tokens=500
            )

            # Clean up in case model wraps JSON in markdown fences
            cleaned = re.sub(
                r"^```(json)?|```$",
                "",
                raw_output,
                flags=re.MULTILINE
            ).strip()

            parsed = json.loads(cleaned)

            clause_type = parsed.get("clause_type", "Unknown")

            # Guard: only accept categories we actually know about
            if clause_type not in self.clause_dictionary and clause_type != "Unknown":
                logger.warning(
                    f"LLM returned unknown category '{clause_type}', falling back to keyword classifier."
                )
                return self.classify_clause(clause_text)

            return clause_type

        except Exception as e:

            logger.error(
                f"LLM classification failed, falling back to keyword classifier: {e}"
            )

            # ---- Improvement 6: hybrid fallback ----
            return self.classify_clause(clause_text)

    # ----------------------------------------------------------------
    # STEP 2 (batch): LLM-BASED CLASSIFICATION (ALL clauses, ONE call)
    # ----------------------------------------------------------------
    def _build_few_shot_batch_prompt(
        self,
        clauses: List[Dict]
    ) -> str:
        """
        Builds a single few-shot prompt that sends ALL extracted
        clauses together, and asks the LLM to return a JSON array
        with one classification per clause (in the same order).

        This replaces N separate ollama.chat() calls with exactly 1 call.
        """

        categories = list(self.clause_dictionary.keys())

        system_rules = f"""You are a strict legal contract clause classifier.

IMPORTANT RULES:
- Use ONLY the given clause texts. Do NOT use external knowledge.
- Classify EACH clause independently using ONLY the text of that clause.
- Choose exactly ONE category per clause from this list: {categories}
- If a clause does not clearly match any category in the list, use "Unknown".
- Do NOT skip any clause and do NOT add extra clauses that are not in the input.
- Return a JSON array with exactly {len(clauses)} objects, in the SAME ORDER as the input clauses.
- Each object must have this exact format: {{"clause_id": <id>, "clause_type": "<category>"}}
- Return ONLY valid JSON. No explanations, no markdown, no extra text.
"""

        example_input = json.dumps(
            [
                {"clause_id": 1, "clause_text": "Either party may terminate this agreement by giving 30 days written notice."},
                {"clause_id": 2, "clause_text": "The client shall pay the vendor within 15 days of invoice receipt."},
                {"clause_id": 3, "clause_text": "This document is for informational purposes only and creates no obligations."}
            ],
            indent=2
        )

        example_output = json.dumps(
            [
                {"clause_id": 1, "clause_type": "Termination"},
                {"clause_id": 2, "clause_type": "Payment"},
                {"clause_id": 3, "clause_type": "Unknown"}
            ],
            indent=2
        )

        # Build the actual input: one entry per extracted clause, with a numeric id
        input_clauses = [
            {"clause_id": idx, "clause_text": clause["text"]}
            for idx, clause in enumerate(clauses)
        ]

        input_json = json.dumps(input_clauses, indent=2)

        prompt = f"""{system_rules}

Example Input:
{example_input}

Example Output:
{example_output}

Now classify the following clauses. Return exactly {len(clauses)} objects, in the same order, using the same clause_id values given below.

Input:
{input_json}

Output:
"""

        return prompt

    def classify_clauses_batch_llm(
        self,
        clauses: List[Dict],
        temperature: float = 0.0
    ) -> List[Dict]:
        """
        Classify ALL clauses in a SINGLE LLM call (instead of one call
        per clause). Uses few-shot prompting with a JSON array input/output.

        If the LLM call fails, or the returned JSON is malformed, or the
        number/order of results doesn't match the input, falls back to
        the keyword-based classifier for every clause (safe default).

        Returns the same `clauses` list, with "clause_type" added/updated
        on each clause dict.
        """

        if not clauses:
            return clauses

        prompt = self._build_few_shot_batch_prompt(clauses)

        try:

            raw_output = chat(
                    user_prompt=prompt,
                    temperature=temperature,
                    max_tokens=2000
            )
            print("\n================ RAW LLM OUTPUT ================\n")
            print(raw_output)
            print("\n================================================\n")

            # Clean up in case model wraps JSON in markdown fences
            cleaned = re.sub(
                r"^```(json)?|```$",
                "",
                raw_output,
                flags=re.MULTILINE
            ).strip()

            parsed = json.loads(cleaned)

            if not isinstance(parsed, list) or len(parsed) != len(clauses):
                raise ValueError(
                    f"Expected a JSON array of {len(clauses)} items, "
                    f"got {len(parsed) if isinstance(parsed, list) else type(parsed)}"
                )

            # Map clause_id -> clause_type from the LLM output
            id_to_type = {}

            for item in parsed:

                clause_id = item.get("clause_id")
                clause_type = item.get("clause_type", "Unknown")

                # Guard: only accept categories we actually know about
                if clause_type not in self.clause_dictionary and clause_type != "Unknown":
                    clause_type = "Unknown"

                id_to_type[clause_id] = clause_type

            # Assign results back to clauses, in original order
            for idx, clause in enumerate(clauses):

                if idx in id_to_type:
                    clause["clause_type"] = id_to_type[idx]
                else:
                    # Missing entry for this clause -> fallback for THIS clause only
                    logger.warning(
                        f"No LLM result for clause_id {idx}, falling back to keyword classifier."
                    )
                    clause["clause_type"] = self.classify_clause(clause["text"])

            return clauses

        except Exception as e:

            logger.error(
                f"Batch LLM classification failed, falling back to keyword classifier for all clauses: {e}"
            )

            # ---- Hybrid fallback: keyword classifier for every clause ----
            for clause in clauses:
                clause["clause_type"] = self.classify_clause(clause["text"])

            return clauses

    # ----------------------------------------------------------------
    # PIPELINE A: Pure regex + keyword (original, fast, offline)
    # ----------------------------------------------------------------
    def process(
        self,
        text: str
    ) -> List[Dict]:
        """
        Original pipeline:
        Regex Extraction -> Keyword Classification
        """

        clauses = self.extract_clauses(
            text
        )

        for clause in clauses:

            clause["clause_type"] = (
                self.classify_clause(
                    clause["text"]
                )
            )

        return clauses

    # ----------------------------------------------------------------
    # PIPELINE B: Hybrid (recommended) -- Regex -> LLM -> Keyword fallback
    # ----------------------------------------------------------------
    def process_hybrid(
        self,
        text: str
    ) -> List[Dict]:
        """
        Recommended industry-standard pipeline:

            Contract
              -> Regex Extraction (fast, deterministic clause boundaries)
              -> Per-clause LLM Classification (accurate semantic labeling)
              -> Keyword Classifier fallback (if LLM fails/unavailable)

        Advantages over feeding the whole contract to the LLM at once:
          - Faster (LLM only classifies, doesn't need to find clause boundaries)
          - Less hallucination (clause text is already grounded in the contract)
          - Better accuracy (LLM focuses on one clause at a time)
          - More robust (keyword fallback if Ollama is down)
        """

        clauses = self.extract_clauses(text)

        for clause in clauses:

            clause["clause_type"] = self.classify_clause_llm(
                clause["text"]
            )

        return clauses

    # ----------------------------------------------------------------
    # PIPELINE C: Batch Hybrid -- Regex -> ONE LLM call for ALL clauses
    # ----------------------------------------------------------------
    def process_hybrid_batch(
        self,
        text: str
    ) -> List[Dict]:
        """
        Optimized pipeline:

            Contract
              -> Regex Extraction (finds all clause boundaries)
              -> ONE single LLM call classifying ALL clauses together
                 (instead of N separate calls, one per clause)
              -> Keyword Classifier fallback (if LLM call fails)

        Advantages over process_hybrid() (N calls):
          - Far fewer API calls: N clauses -> 1 call instead of N calls
          - Lower latency overall (no per-call network/model load overhead)
          - Still grounded (clause boundaries already fixed by regex,
            so the LLM only classifies, doesn't need to invent clauses)
        """

        clauses = self.extract_clauses(text)

        clauses = self.classify_clauses_batch_llm(clauses)

        return clauses


# --------------------------------------------------
# TESTING
# --------------------------------------------------

if __name__ == "__main__":

    sample_text = """
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

    extractor = ClauseExtractor(
        "data/clauses.json"
    )

    # ---- Pipeline A: Regex + Keyword (original, fast, fully offline) ----
    result = extractor.process(
        sample_text
    )

    print("=== Pipeline A: Regex + Keyword classification ===")
    print(
        json.dumps(
            result,
            indent=4
        )
    )

    # ---- Pipeline B: Hybrid Regex + LLM (recommended) ----
    hybrid_result = extractor.process_hybrid(
        sample_text
    )

    print("\n=== Pipeline B: Hybrid (Regex -> LLM per-clause -> Keyword fallback) ===")
    print(
        json.dumps(
            hybrid_result,
            indent=4
        )
    )

    # ---- Pipeline C: Batch Hybrid (Regex -> ONE LLM call for all clauses) ----
    batch_result = extractor.process_hybrid_batch(
        sample_text
    )

    print("\n=== Pipeline C: Batch Hybrid (Regex -> 1 LLM call -> Keyword fallback) ===")
    print(
        json.dumps(
            batch_result,
            indent=4
        )
    )