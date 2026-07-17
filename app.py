"""
app.py — Contract Risk Analyzer API (RESTful, versioned)

RESTful architecture under /api/v1:

  POST   /api/v1/risk-analyze            → run full pipeline, returns a report (with id)
  GET    /api/v1/risk-report/{id}        → retrieve a previously generated report
  GET    /api/v1/risk-reports            → list all stored report ids

  GET    /api/v1/risk-patterns           → list all risk patterns
  GET    /api/v1/risk-patterns/{id}      → get a single risk pattern
  POST   /api/v1/risk-patterns           → create a new risk pattern
  PUT    /api/v1/risk-patterns/{id}      → update an existing risk pattern
  DELETE /api/v1/risk-patterns/{id}      → delete a risk pattern

  GET    /version                        → API version info
  GET    /api/v1/health                  → comprehensive health check
  GET    /                                → serves frontend (index.html)

Run with:
    python3 app.py
Interactive docs:
    http://127.0.0.1:8080/docs
"""

import json
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import evaluation
from text_preprocessing import TextPreprocessor
from clause_extraction import ClauseExtractor
from risk import RiskDetector
from SummaryGenerator import SummaryGenerator
from evaluation import run_pipeline_evaluation

RISK_PATTERNS_PATH = "data/important_risk_patterns.json"
VALID_LEVELS = {"HIGH", "MEDIUM", "LOW"}

CLAUSE_JSON = Path("data/clauses.json")
RISK_JSON = Path(RISK_PATTERNS_PATH)


# --------------------------------------------------
# App setup
# --------------------------------------------------
app = FastAPI(
    title="Contract Risk Analyzer API",
    description="RESTful NLP pipeline for legal contract clause extraction, risk detection, and summarization.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Boot pipeline modules once at startup
# --------------------------------------------------
preprocessor      = TextPreprocessor()
clause_extractor  = ClauseExtractor("data/clauses.json")
risk_detector     = RiskDetector(RISK_PATTERNS_PATH)
summary_generator = SummaryGenerator()

# In-memory store for generated reports (POST /risk-analyze → GET /risk-report/{id})
REPORTS: dict = {}


# --------------------------------------------------
# Schemas
# --------------------------------------------------
class AnalyzeRequest(BaseModel):
    text: str
    risk_level: Optional[str] = None   # "HIGH" | "MEDIUM" | "LOW"
    top_n: Optional[int] = None


class RiskPattern(BaseModel):
    pattern: str
    risk: str
    level: str


class RiskPatternUpdate(BaseModel):
    pattern: Optional[str] = None
    risk: Optional[str] = None
    level: Optional[str] = None


# --------------------------------------------------
# Helpers — risk pattern persistence
# --------------------------------------------------
def load_patterns() -> List[dict]:
    with open(RISK_PATTERNS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_patterns(patterns: List[dict]) -> None:
    with open(RISK_PATTERNS_PATH, "w", encoding="utf-8") as f:
        json.dump(patterns, f, indent=2)

    # Reload the detector's in-memory rules so new patterns take effect immediately
    risk_detector.rules = patterns


def ensure_ids(patterns: List[dict]) -> List[dict]:
    """Give every pattern a stable id if it doesn't already have one."""
    changed = False
    for p in patterns:
        if "id" not in p:
            p["id"] = str(uuid.uuid4())[:8]
            changed = True
    if changed:
        save_patterns(patterns)
    return patterns


# --------------------------------------------------
# Helper — run the full analysis pipeline
# --------------------------------------------------
def run_pipeline(raw_text: str) -> dict:
    cleaned = preprocessor.preprocess(raw_text)
    clauses = clause_extractor.process(cleaned)

    for clause in clauses:
        clause["risks"] = risk_detector.detect_risks(clause["text"])

    summary = summary_generator.generate_executive_summary(clauses)
    print("Summary generated")
    evaluation = run_pipeline_evaluation(
    original_contract=raw_text,
    clauses=clauses,
    summary=summary,
    clause_dictionary=clause_extractor.clause_dictionary,
    risk_patterns=risk_detector.rules

    )
    print("Evaluation returned:")
    print(evaluation)
    return {
        "cleaned_text": cleaned,
        "clauses": clauses,
        "summary": summary,
        "evaluation": evaluation
    }


# ==================================================
# ROOT / META ROUTES
# ==================================================
@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


@app.get("/version")
def version():
    return {
        "name": "Contract Risk Analyzer API",
        "version": "1.0.0",
        "framework": "FastAPI",
        "api_base": "/api/v1",
    }


@app.get(
    "/api/v1/health",
    tags=["Health"],
    summary="Comprehensive Health Check",
    description="Checks API availability and validates all critical components."
)
def health_check():
    try:
        checks = {
            "preprocessor_loaded":
                preprocessor is not None,
            "clause_extractor_loaded":
                clause_extractor is not None,
            "risk_detector_loaded":
                risk_detector is not None,
            "summary_generator_loaded":
                summary_generator is not None,
            "clause_dictionary_file_exists":
                CLAUSE_JSON.exists(),
            "risk_patterns_file_exists":
                RISK_JSON.exists()
        }
        failed_components = [
            component
            for component, status in checks.items()
            if not status
        ]
        # Service unavailable
        if failed_components:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "unhealthy",
                    "message":
                        "One or more required components are unavailable",
                    "failed_components":
                        failed_components,
                    "checks":
                        checks
                }
            )
        # Success response
        return {
            "status": "healthy",
            "message":
                "AI Contract Risk Analyzer API is running successfully",
            "version": "v1",
            "checks": checks
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message":
                    "Unexpected error occurred during health check",
                "error":
                    str(exc)
            }
        )


# ==================================================
# POST /api/v1/risk-analyze  — submit contract text, get a report
# ==================================================
@app.post("/api/v1/risk-analyze", status_code=201)
def risk_analyze(req: AnalyzeRequest):
    raw_text = req.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="No contract text provided.")

    risk_level = req.risk_level.upper() if req.risk_level else None
    top_n = req.top_n

    if risk_level and risk_level not in VALID_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid risk_level. Choose from {VALID_LEVELS}",
        )

    if top_n is not None and top_n < 1:
        raise HTTPException(status_code=400, detail="top_n must be >= 1")

    try:
        result = run_pipeline(raw_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

    # Flatten risks across all clauses
    all_risks = []
    for clause in result["clauses"]:
        for risk in clause.get("risks", []):
            all_risks.append({
                "clause_number": clause["clause_number"],
                "clause_title": clause["clause_title"],
                "clause_type": clause["clause_type"],
                **risk,
            })

    # Default: nothing specified → top 5 HIGH
    if risk_level is None and top_n is None:
        filtered = [r for r in all_risks if r["level"] == "HIGH"][:5]
        filter_applied = "default (top 5 HIGH)"
    else:
        filtered = all_risks
        if risk_level:
            filtered = [r for r in filtered if r["level"] == risk_level]
        if top_n:
            filtered = filtered[:top_n]
        filter_applied = f"level={risk_level or 'ALL'}, top_n={top_n or 'ALL'}"

    result["filtered_risks"] = filtered
    result["filter_applied"] = filter_applied

    # Store report so it can be retrieved later via GET /risk-report/{id}
    report_id = str(uuid.uuid4())[:8]
    result["id"] = report_id
    REPORTS[report_id] = result

    return result


# ==================================================
# GET /api/v1/risk-reports — list all stored report ids
# ==================================================
@app.get("/api/v1/risk-reports")
def list_reports():
    return {
        "count": len(REPORTS),
        "report_ids": list(REPORTS.keys()),
    }


# ==================================================
# GET /api/v1/risk-report/{id} — retrieve one report
# ==================================================
@app.get("/api/v1/risk-report/{report_id}")
def get_risk_report(report_id: str):
    report = REPORTS.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"No report found with id '{report_id}'")
    return report


# ==================================================
# GET /api/v1/risk-patterns — list all patterns
# ==================================================
@app.get("/api/v1/risk-patterns")
def list_risk_patterns():
    patterns = ensure_ids(load_patterns())
    return {"count": len(patterns), "patterns": patterns}


# ==================================================
# GET /api/v1/risk-patterns/{id} — get one pattern
# ==================================================
@app.get("/api/v1/risk-patterns/{pattern_id}")
def get_risk_pattern(pattern_id: str):
    patterns = ensure_ids(load_patterns())
    for p in patterns:
        if p["id"] == pattern_id:
            return p
    raise HTTPException(status_code=404, detail=f"No risk pattern found with id '{pattern_id}'")


# ==================================================
# POST /api/v1/risk-patterns — create a new pattern
# ==================================================
@app.post("/api/v1/risk-patterns", status_code=201)
def create_risk_pattern(new_pattern: RiskPattern):
    if new_pattern.level.upper() not in VALID_LEVELS:
        raise HTTPException(status_code=400, detail=f"level must be one of {VALID_LEVELS}")

    patterns = ensure_ids(load_patterns())

    entry = new_pattern.dict()
    entry["level"] = entry["level"].upper()
    entry["id"] = str(uuid.uuid4())[:8]

    patterns.append(entry)
    save_patterns(patterns)

    return entry


# ==================================================
# PUT /api/v1/risk-patterns/{id} — update a pattern
# ==================================================
@app.put("/api/v1/risk-patterns/{pattern_id}")
def update_risk_pattern(pattern_id: str, update: RiskPatternUpdate):
    patterns = ensure_ids(load_patterns())

    for p in patterns:
        if p["id"] == pattern_id:
            if update.pattern is not None:
                p["pattern"] = update.pattern
            if update.risk is not None:
                p["risk"] = update.risk
            if update.level is not None:
                if update.level.upper() not in VALID_LEVELS:
                    raise HTTPException(status_code=400, detail=f"level must be one of {VALID_LEVELS}")
                p["level"] = update.level.upper()

            save_patterns(patterns)
            return p

    raise HTTPException(status_code=404, detail=f"No risk pattern found with id '{pattern_id}'")


# ==================================================
# DELETE /api/v1/risk-patterns/{id} — remove a pattern
# ==================================================
@app.delete("/api/v1/risk-patterns/{pattern_id}", status_code=200)
def delete_risk_pattern(pattern_id: str):
    patterns = ensure_ids(load_patterns())

    remaining = [p for p in patterns if p["id"] != pattern_id]

    if len(remaining) == len(patterns):
        raise HTTPException(status_code=404, detail=f"No risk pattern found with id '{pattern_id}'")

    save_patterns(remaining)
    return {"message": f"Risk pattern '{pattern_id}' deleted.", "remaining_count": len(remaining)}


# --------------------------------------------------
# Run directly with: python3 app.py
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    print("\n Contract Risk Analyzer API (RESTful, FastAPI)")
    print(" ───────────────────────────────────────────────")
    print("  POST   /api/v1/risk-analyze            → submit contract, get report")
    print("  GET    /api/v1/risk-report/{id}         → retrieve a report")
    print("  GET    /api/v1/risk-reports             → list all report ids")
    print("  GET    /api/v1/risk-patterns            → list risk patterns")
    print("  GET    /api/v1/risk-patterns/{id}       → get one pattern")
    print("  POST   /api/v1/risk-patterns            → create pattern")
    print("  PUT    /api/v1/risk-patterns/{id}       → update pattern")
    print("  DELETE /api/v1/risk-patterns/{id}       → delete pattern")
    print("  GET    /version  |  /api/v1/health  |  /docs")
    print(" ───────────────────────────────────────────────\n")
if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(
    "app:app",
    host="0.0.0.0",
    port=int(os.getenv("PORT", 8080)),
    reload=False,
    ) 
