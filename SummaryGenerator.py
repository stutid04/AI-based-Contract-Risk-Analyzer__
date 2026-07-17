from collections import Counter
import json
import time
from llm_client import chat



class SummaryGenerator:

    def __init__(self, use_llm=True):
        self.use_llm = use_llm

    # ==================================================
    # MAIN ENTRY POINT
    # ==================================================

    def generate_executive_summary(self, clauses):

     print("\n" + "=" * 70)
     print("SUMMARY GENERATION STARTED")
     print("=" * 70)

     if self.use_llm:

        try:

            summary = self._generate_llm_summary(
                clauses
            )

            print(
                "\n[SUCCESS] LLM Summary Generated Successfully"
            )

            print("=" * 70)
            print("SUMMARY GENERATION COMPLETED")
            print("=" * 70)

            return summary

        except Exception as e:

            print(
                "\n[ERROR] LLM SUMMARY FAILED"
            )

            print(
                f"[ERROR DETAILS] {e}"
            )

            print(
                "\n[INFO] Switching to Rule-Based Summary..."
            )

     summary = self._generate_rule_based_summary(
        clauses
     )

     print(
        "\n[SUCCESS] Rule-Based Summary Generated"
     )

     print("=" * 70)
     print("SUMMARY GENERATION COMPLETED")
     print("=" * 70)

     return summary

    # ==================================================
    # LLM SUMMARY GENERATION
    # ==================================================

    def _generate_llm_summary(self,clauses):
        print(
        "\n[INFO] Generating summary using Llama 3.2"
        )

        start_time = time.time()

        clause_types = set()
        high_risks = []
        medium_risks = []
        low_risks = []

        for clause in clauses:
            clause_types.add(
                clause["clause_type"]
            )

            for risk in clause.get(
                "risks",
                []
            ):

                if risk["level"] == "HIGH":

                    high_risks.append(
                        risk["risk"]
                    )

                elif risk["level"] == "MEDIUM":

                    medium_risks.append(
                        risk["risk"]
                    )

                elif risk["level"] == "LOW":

                    low_risks.append(
                        risk["risk"]
                    )

        if high_risks:
            overall_risk = "HIGH"

        elif medium_risks:
            overall_risk = "MEDIUM"

        else:
            overall_risk = "LOW"

        structured_input = {

            "overall_risk_level":
                overall_risk,

            "clause_categories":
                sorted(
                    list(
                        clause_types
                    )
                ),

            "high_risks":
                sorted(
                    list(
                        set(
                            high_risks
                        )
                    )
                ),

            "medium_risks":
                sorted(
                    list(
                        set(
                            medium_risks
                        )
                    )
                ),

            "low_risks":
                sorted(
                    list(
                        set(
                            low_risks
                        )
                    )
                ),

            "risk_counts": {

                "high":
                    len(
                        high_risks
                    ),

                "medium":
                    len(
                        medium_risks
                    ),

                "low":
                    len(
                        low_risks
                    )
            }
        }

        print(
            "\n[INFO] Structured Input Sent To openRouter:"
        )

        print(
            json.dumps(
                structured_input,
                indent=4
            )
        )

        prompt = f"""
    You are an experienced legal contract reviewer.

    Based solely on the provided clause categories and detected risks, generate a concise executive summary.

    Writing style requirements:

    - Professional
    - Formal
    - Business-oriented
    - Easy for non-technical stakeholders to understand

    Content requirements:
    
    1. Provide an overall assessment of the contract.
    2. Mention major clause categories present in the contract.
    3. Mention critical high-risk findings.
    4. Mention notable medium-risk findings if present.
    5. Keep the summary under 250 words.
    6. Use paragraphs  and also sub bullet points if needed for clarity.
    9. No markdown.
    10. No assumptions.
    11. Do not invent risks.
    12. Do not invent clause categories.
    13. Use only the supplied information.
    14. the summary should be clear, concise.
    Input:

    {json.dumps(structured_input, indent=2)}

    Generate only the executive summary text.
    """

        print(
            "\n[INFO] Sending Request To OpenRouter..."
        )

        summary = chat(
                user_prompt=prompt,
                temperature=0.1,
                max_tokens=200,
        )
        if not summary:

            raise RuntimeError(
                "Empty response received from OpenRouter."
            )

        print(
            "\n[INFO] Response Received From OpenRouter"
        )

        print(
            "\n========== OPENROUTER SUMMARY =========="
        )

        print(
            summary
        )

        print(
            "================================="
        )

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"\n[INFO] OpenRouter Generation Time: "
            f"{elapsed:.2f} seconds"
        )

        return summary

    # ==================================================
    # RULE-BASED FALLBACK
    # (UNCHANGED)
    # ==================================================

    def _generate_rule_based_summary(self, clauses):
        print(
            "\n[INFO] USING RULE-BASED FALLBACK SUMMARY"
        )

        clause_types = set()
        risk_counter = Counter()

        high_risks = []
        medium_risks = []
        low_risks = []

        for clause in clauses:

            clause_types.add(
                clause["clause_type"]
            )

            for risk in clause.get("risks", []):

                level = risk["level"]

                risk_counter[level] += 1

                if level == "HIGH":

                    high_risks.append(
                        risk["risk"]
                    )

                elif level == "MEDIUM":

                    medium_risks.append(
                        risk["risk"]
                    )

                elif level == "LOW":

                    low_risks.append(
                        risk["risk"]
                    )

        # Overall Risk

        if risk_counter["HIGH"] > 0:

            overall_risk = "HIGH"

        elif risk_counter["MEDIUM"] > 0:

            overall_risk = "MEDIUM"

        else:

            overall_risk = "LOW"

        summary = []

        summary.append(
            "EXECUTIVE SUMMARY\n"
        )

        summary.append(
            f"The agreement contains "
            f"{len(clause_types)} clause categories: "
            f"{', '.join(sorted(clause_types))}.\n"
        )

        if high_risks:

            summary.append(
                "High-Risk Provisions:"
            )

            for risk in set(high_risks):

                summary.append(
                    f"• {risk}"
                )

            summary.append("")

        if medium_risks:

            summary.append(
                "Medium-Risk Provisions:"
            )

            for risk in set(medium_risks):

                summary.append(
                    f"• {risk}"
                )

            summary.append("")

        summary.append(
            "Risk Distribution:"
        )

        summary.append(
            f"• High Risks: "
            f"{risk_counter['HIGH']}"
        )

        summary.append(
            f"• Medium Risks: "
            f"{risk_counter['MEDIUM']}"
        )

        summary.append(
            f"• Low Risks: "
            f"{risk_counter['LOW']}"
        )

        summary.append("")

        summary.append(
            f"Overall Contract Risk Level: "
            f"{overall_risk}"
        )

        return "\n".join(summary)