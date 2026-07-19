"""
clause_extraction.py

Clause Extraction and Classification Module
(Hybrid architecture: Regex extraction -> Per-clause LLM classification
 -> Keyword-based fallback classifier, using local Llama 3.2 via Ollama)

Supports ALL agreement types:
  - Service Agreements, License Agreements, Employment Contracts,
    NDAs, SaaS Agreements, Franchise Agreements, Lease Agreements,
    Partnership Agreements, Purchase Agreements, and more.
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
    Extracts and classifies clauses from legal contracts of ANY type.
    """

    def __init__(
        self,
        clause_json_path: str,
        threshold: int = 1,
        ollama_model: str = "llama3.2"
    ):
        self.threshold = threshold
        self.ollama_model = ollama_model

        try:
            with open(clause_json_path, "r", encoding="utf-8") as file:
                self.clause_dictionary = json.load(file)
        except Exception as e:
            logger.error(f"Unable to load JSON file: {e}")
            raise

    # ----------------------------------------------------------------
    # STEP 1: REGEX-BASED EXTRACTION
    # ----------------------------------------------------------------
    def extract_clauses(self, text: str) -> List[Dict]:
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
                re.IGNORECASE | re.MULTILINE | re.VERBOSE
            )

            matches = list(clause_pattern.finditer(text))
            clauses = []

            if not matches:
                return clauses

            for i in range(len(matches)):
                start = matches[i].start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                block = text[start:end].strip()
                lines = block.splitlines()

                if not lines:
                    continue

                heading = lines[0].strip()
                clause_number = ""
                clause_title = heading

                number_match  = re.match(r'^(\d+(?:\.\d+)*\.?)', heading)
                section_match = re.match(r'^(section\s+\d+)', heading, re.IGNORECASE)
                article_match = re.match(r'^(article\s+[ivxlcdm]+)', heading, re.IGNORECASE)
                alpha_match   = re.match(r'^([A-Z]\.)', heading)
                bracket_match = re.match(r'^(\([a-z]\))', heading)

                if number_match:
                    clause_number = number_match.group(1)
                    clause_title  = heading[len(clause_number):].strip()
                elif section_match:
                    clause_number = section_match.group(1)
                    clause_title  = heading
                elif article_match:
                    clause_number = article_match.group(1)
                    clause_title  = heading
                elif alpha_match:
                    clause_number = alpha_match.group(1)
                    clause_title  = heading[len(clause_number):].strip()
                elif bracket_match:
                    clause_number = bracket_match.group(1)
                    clause_title  = heading[len(clause_number):].strip()

                clauses.append({
                    "clause_number": clause_number,
                    "clause_title":  clause_title,
                    "text":          block
                })

            return clauses

        except Exception as e:
            logger.error(f"Clause extraction failed: {e}")
            return []

    # ----------------------------------------------------------------
    # STEP 2 (fallback): KEYWORD-BASED CLASSIFICATION — UNCHANGED
    # ----------------------------------------------------------------
    def classify_clause(self, clause_text: str) -> str:
        try:
            clause_text_lower = clause_text.lower()
            scores = {}

            for category, keywords in self.clause_dictionary.items():
                score = 0
                for keyword in keywords:
                    score += clause_text_lower.count(keyword.lower())
                scores[category] = score

            best_category = max(scores, key=scores.get, default="Unknown")

            if scores[best_category] < self.threshold:
                return "Unknown"

            return best_category

        except Exception as e:
            logger.error(f"Keyword classification failed: {e}")
            return "Unknown"

    # ----------------------------------------------------------------
    # STEP 2 (primary): PER-CLAUSE LLM PROMPT
    # Updated: generic for ALL agreement types
    # ----------------------------------------------------------------
    def _build_few_shot_classification_prompt(self, clause_text: str) -> str:
        """
        Few-shot prompt to classify a SINGLE clause from ANY agreement type
        (Service, License, Employment, NDA, SaaS, Franchise, Lease, etc.).
        """

        categories = list(self.clause_dictionary.keys())

        system_rules = f"""You are a strict legal contract clause classifier.
You work with ALL types of legal agreements including but not limited to:
Service Agreements, License Agreements, Employment Contracts, NDAs,
SaaS Agreements, Franchise Agreements, Lease Agreements, Partnership
Agreements, and Purchase Agreements.

IMPORTANT RULES:
- Use ONLY the given clause text. Do NOT use external knowledge.
- Identify the clause type based on its LEGAL PURPOSE and CONTENT,
  regardless of which type of agreement it appears in.
- Choose exactly ONE category from this list: {categories}
- If the clause does not clearly match any category in the list,
  return "Unknown". Do NOT invent new categories.
- Return ONLY valid JSON in the exact format shown. No explanations, no markdown.
"""

        # Few-shot examples covering diverse agreement types
        examples = [
            # Service Agreement
            (
                "The Service Provider shall deliver software development services "
                "as described in Schedule A within the agreed timeline.",
                {"clause_type": "Services"}
            ),
            # License Agreement
            (
                "Licensor grants Licensee a non-exclusive, non-transferable license "
                "to use the Software solely for internal business purposes.",
                {"clause_type": "Royalty"}
            ),
            # Employment Contract
            (
                "Employee shall receive an annual salary of $80,000 payable in "
                "bi-weekly instalments, subject to applicable deductions.",
                {"clause_type": "Compensation"}
            ),
            # NDA
            (
                "Each party agrees to keep confidential all proprietary information "
                "disclosed by the other party and not to disclose it to third parties.",
                {"clause_type": "Confidentiality"}
            ),
            # Termination clause (any agreement type)
            (
                "Either party may terminate this agreement upon 30 days written notice "
                "or immediately upon material breach by the other party.",
                {"clause_type": "Termination"}
            ),
            # Non-matching clause -> Unknown
            (
                "This document is provided for informational purposes only "
                "and does not constitute a legally binding obligation.",
                {"clause_type": "Unknown"}
            ),
        ]

        examples_text = ""
        for i, (inp, out) in enumerate(examples, 1):
            examples_text += f'\nExample {i}:\nClause: "{inp}"\nOutput: {json.dumps(out)}\n'

        prompt = f"""{system_rules}
{examples_text}
Now classify the following clause from a legal agreement.

Clause: "{clause_text}"
Output:
"""
        return prompt

    def classify_clause_llm(self, clause_text: str, temperature: float = 0.0) -> str:
        prompt = self._build_few_shot_classification_prompt(clause_text)

        try:
            raw_output = chat(
                user_prompt=prompt,
                temperature=temperature,
                max_tokens=500
            )

            cleaned = re.sub(
                r"^```(json)?|```$", "", raw_output, flags=re.MULTILINE
            ).strip()

            parsed = json.loads(cleaned)
            clause_type = parsed.get("clause_type", "Unknown")

            if clause_type not in self.clause_dictionary and clause_type != "Unknown":
                logger.warning(
                    f"LLM returned unknown category '{clause_type}', "
                    f"falling back to keyword classifier."
                )
                return self.classify_clause(clause_text)

            return clause_type

        except Exception as e:
            logger.error(
                f"LLM classification failed, falling back to keyword classifier: {e}"
            )
            return self.classify_clause(clause_text)

    # ----------------------------------------------------------------
    # BATCH LLM PROMPT
    # Updated: generic for ALL agreement types
    # ----------------------------------------------------------------
    def _build_few_shot_batch_prompt(self, clauses: List[Dict]) -> str:
        """
        Single few-shot prompt that sends ALL extracted clauses together
        and asks the LLM to classify each one in ONE call.
        Works for ALL agreement types.
        """

        categories = list(self.clause_dictionary.keys())

        system_rules = f"""You are a strict legal contract clause classifier.
You work with ALL types of legal agreements including but not limited to:
Service Agreements, License Agreements, Employment Contracts, NDAs,
SaaS Agreements, Franchise Agreements, Lease Agreements, Partnership
Agreements, and Purchase Agreements.

IMPORTANT RULES:
- Use ONLY the given clause texts. Do NOT use external knowledge.
- Classify EACH clause independently based on its LEGAL PURPOSE and CONTENT,
  regardless of which type of agreement it appears in.
- Choose exactly ONE category per clause from this list: {categories}
- If a clause does not clearly match any category, use "Unknown".
- Do NOT skip any clause. Do NOT add extra clauses not in the input.
- Return a JSON array with exactly {len(clauses)} objects, in the SAME ORDER as input.
- Each object must have this exact format: {{"clause_id": <id>, "clause_type": "<category>"}}
- Return ONLY valid JSON. No explanations, no markdown, no extra text.
"""

        # Few-shot example covering diverse agreement types in batch format
        example_input = json.dumps([
            {
                "clause_id": 0,
                "clause_text": (
                    "Licensor grants Licensee a non-exclusive, "
                    "non-transferable license to use the Software."
                )
            },
            {
                "clause_id": 1,
                "clause_text": (
                    "Employee shall receive an annual salary of $80,000 "
                    "payable in bi-weekly instalments."
                )
            },
            {
                "clause_id": 2,
                "clause_text": (
                    "Each party shall keep confidential all proprietary "
                    "information disclosed by the other party."
                )
            },
            {
                "clause_id": 3,
                "clause_text": (
                    "Either party may terminate this agreement upon "
                    "30 days written notice."
                )
            },
            {
                "clause_id": 4,
                "clause_text": (
                    "This document is for informational purposes only "
                    "and creates no obligations."
                )
            },
        ], indent=2)

        example_output = json.dumps([
            {"clause_id": 0, "clause_type": "Royalty"},
            {"clause_id": 1, "clause_type": "Compensation"},
            {"clause_id": 2, "clause_type": "Confidentiality"},
            {"clause_id": 3, "clause_type": "Termination"},
            {"clause_id": 4, "clause_type": "Unknown"},
        ], indent=2)

        # Actual clauses to classify
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

Now classify the following clauses from a legal agreement.
Return exactly {len(clauses)} objects in the same order, using the same clause_id values.

Input:
{input_json}

Output:
"""
        return prompt

    def classify_clauses_batch_llm(
        self, clauses: List[Dict], temperature: float = 0.0
    ) -> List[Dict]:
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

            cleaned = re.sub(
                r"^```(json)?|```$", "", raw_output, flags=re.MULTILINE
            ).strip()

            parsed = json.loads(cleaned)

            if not isinstance(parsed, list) or len(parsed) != len(clauses):
                raise ValueError(
                    f"Expected a JSON array of {len(clauses)} items, "
                    f"got {len(parsed) if isinstance(parsed, list) else type(parsed)}"
                )

            id_to_type = {}
            for item in parsed:
                clause_id   = item.get("clause_id")
                clause_type = item.get("clause_type", "Unknown")
                if clause_type not in self.clause_dictionary and clause_type != "Unknown":
                    clause_type = "Unknown"
                id_to_type[clause_id] = clause_type

            for idx, clause in enumerate(clauses):
                if idx in id_to_type:
                    clause["clause_type"] = id_to_type[idx]
                else:
                    logger.warning(
                        f"No LLM result for clause_id {idx}, "
                        f"falling back to keyword classifier."
                    )
                    clause["clause_type"] = self.classify_clause(clause["text"])

            return clauses

        except Exception as e:
            logger.error(
                f"Batch LLM classification failed, "
                f"falling back to keyword classifier for all clauses: {e}"
            )
            for clause in clauses:
                clause["clause_type"] = self.classify_clause(clause["text"])
            return clauses

    # ----------------------------------------------------------------
    # PIPELINE A: Regex + Keyword (fully offline)
    # ----------------------------------------------------------------
    def process(self, text: str) -> List[Dict]:
        clauses = self.extract_clauses(text)
        for clause in clauses:
            clause["clause_type"] = self.classify_clause(clause["text"])
        return clauses

    # ----------------------------------------------------------------
    # PIPELINE B: Regex -> Per-clause LLM -> Keyword fallback
    # ----------------------------------------------------------------
    def process_hybrid(self, text: str) -> List[Dict]:
        clauses = self.extract_clauses(text)
        for clause in clauses:
            clause["clause_type"] = self.classify_clause_llm(clause["text"])
        return clauses

    # ----------------------------------------------------------------
    # PIPELINE C: Regex -> ONE batch LLM call -> Keyword fallback
    # ----------------------------------------------------------------
    def process_hybrid_batch(self, text: str) -> List[Dict]:
        clauses = self.extract_clauses(text)
        clauses = self.classify_clauses_batch_llm(clauses)
        return clauses


# --------------------------------------------------
# TESTING
# --------------------------------------------------

if __name__ == "__main__":

    # Sample covering multiple agreement types
    sample_text = """
1. SERVICES

The Service Provider shall deliver software development and consulting
services as described in Schedule A.

1.1 LICENSE GRANT

Licensor grants Licensee a non-exclusive, non-transferable license
to use the Software for internal business purposes only.

SECTION 2

All confidential information disclosed by either party shall remain
confidential for a period of five years.

ARTICLE III

Employee shall receive an annual salary of $80,000 payable bi-weekly.

A. TERMINATION

Either party may terminate this agreement upon 30 days written notice.

(a) ROYALTIES

Licensee shall pay a royalty of 5% of net revenues derived from the Software.
"""

    extractor = ClauseExtractor("data/clauses.json")

    print("=== Pipeline A: Regex + Keyword ===")
    print(json.dumps(extractor.process(sample_text), indent=4))

    print("\n=== Pipeline B: Per-clause LLM ===")
    print(json.dumps(extractor.process_hybrid(sample_text), indent=4))

    print("\n=== Pipeline C: Batch LLM (1 call) ===")
    print(json.dumps(extractor.process_hybrid_batch(sample_text), indent=4))
