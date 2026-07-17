"""
AI Contract Risk Analyzer — Streamlit Frontend (Enhanced v3 — Tabbed Layout)
Connects to the FastAPI backend at http://127.0.0.1:8080/api/v1

UI Enhancement only — all backend logic, API calls, session state,
filtering, and chart data are preserved exactly from v2.
"""

import io
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Contract Risk Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# API_BASE = "http://127.0.0.1:8080/api/v1"
import os

API_BASE = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8080/api/v1"
)

# ─── Clause description mapping (unchanged from v2) ───────────────────────────
CLAUSE_DESCRIPTIONS = {
    "Termination":                             "Clauses governing termination rights, conditions, and procedures.",
    "Confidentiality":                         "Clauses protecting confidential information and disclosure obligations.",
    "Payment":                                 "Clauses defining payment obligations, schedules, and timelines.",
    "Compensation":                            "Clauses covering compensation, fees, and financial arrangements.",
    "Indemnification":                         "Clauses specifying indemnification duties and liability exposure.",
    "Miscellaneous":                           "General miscellaneous provisions and boilerplate terms.",
    "Notices":                                 "Clauses governing formal notice requirements and communication.",
    "Investor":                                "Clauses relating to investor rights, obligations, and protections.",
    "Agent Services":                          "Clauses defining agent roles, duties, and service scope.",
    "Expenses And Fees":                       "Clauses covering expense reimbursement, fees, and cost allocation.",
    "Term":                                    "Clauses defining the duration, renewal, and expiration of the agreement.",
    "Royalty":                                 "Clauses covering royalty payments, schedules, and guarantees.",
    "Definitions":                             "Clauses providing definitions of key terms used in the contract.",
    "Rights And Obligations Upon Termination": "Clauses specifying rights and duties that survive termination.",
    "Independent Contractors":                 "Clauses establishing the independent contractor relationship.",
    "Services":                                "Clauses outlining the scope of services to be provided.",
}

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

/* ── Design tokens ── */
:root {
    --navy:      #0F1E36;
    --slate:     #1C3358;
    --blue:      #2A5C99;
    --accent:    #3B82F6;
    --gold:      #E8B84B;
    --surface:   #F1F5F9;
    --card:      #FFFFFF;
    --text:      #1A202C;
    --muted:     #64748B;
    --border:    #E2E8F0;
    --red:       #DC2626;
    --amber:     #D97706;
    --green:     #16A34A;
    --red-bg:    #FEF2F2;
    --amber-bg:  #FFFBEB;
    --green-bg:  #F0FDF4;
}

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); }
.main { background: var(--surface); }
.block-container { padding: 0 2rem 4rem 2rem !important; max-width: 1340px; }

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, var(--navy) 0%, var(--slate) 60%, #1e4a7a 100%);
    border-radius: 0 0 24px 24px;
    padding: 2.4rem 3rem;
    margin: -1rem -2rem 2.5rem -2rem;
    display: flex; align-items: center; gap: 2rem;
    position: relative; overflow: hidden;
}
.app-header::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.app-header::after {
    content: ''; position: absolute; bottom: -40px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(232,184,75,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.logo-mark {
    width: 66px; height: 66px;
    background: linear-gradient(135deg, var(--gold), #f59e0b);
    border-radius: 16px; display: flex; align-items: center; justify-content: center;
    font-size: 1.9rem; box-shadow: 0 8px 24px rgba(232,184,75,0.35);
    flex-shrink: 0; position: relative; z-index: 1;
}
.header-text { position: relative; z-index: 1; flex: 1; }
.header-title {
    font-family: 'DM Serif Display', serif; font-size: 2rem;
    color: #fff; margin: 0; line-height: 1.15; letter-spacing: -0.4px;
}
.header-subtitle {
    font-size: 0.92rem; color: rgba(255,255,255,0.62);
    margin: 0.38rem 0 0 0; font-weight: 400; max-width: 540px; line-height: 1.5;
}
.header-badge {
    position: relative; z-index: 1;
    background: rgba(59,130,246,0.18); border: 1px solid rgba(59,130,246,0.38);
    color: #93C5FD; font-size: 0.75rem; font-weight: 600;
    padding: 0.32rem 0.85rem; border-radius: 20px; letter-spacing: 0.6px;
    text-transform: uppercase; white-space: nowrap; align-self: flex-start; margin-top: 6px;
}

/* ── Input section ── */
.input-panel {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; padding: 1.6rem 1.75rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 1.25rem;
}
.input-panel-title {
    font-size: 0.78rem; font-weight: 700; color: var(--navy);
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 1.1rem;
    display: flex; align-items: center; gap: 0.4rem;
}
.input-label {
    font-size: 0.78rem; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.7px; margin-bottom: 0.45rem;
}
.col-divider {
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; color: var(--muted); font-weight: 600;
    letter-spacing: 0.5px; padding-top: 3.5rem;
}

/* ── Analyze button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #1d4ed8 100%) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 0.72rem 2.5rem !important;
    font-size: 0.97rem !important; font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 14px rgba(59,130,246,0.32) !important;
    transition: all 0.2s ease !important; width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.44) !important;
}

/* ── Upload widget ── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border); border-radius: 12px;
    padding: 0.4rem; background: #FAFBFC; transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent); }

/* ── Success / alert ── */
.stAlert { border-radius: 10px !important; }

/* ── Tabs — override Streamlit defaults ── */
[data-baseweb="tab-list"] {
    gap: 0px !important;
    background: var(--card) !important;
    border-bottom: 2px solid var(--border) !important;
    padding: 0 0.5rem !important;
    border-radius: 12px 12px 0 0 !important;
    margin-top: 1.25rem !important;
}
[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.87rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.25rem !important;
    color: var(--muted) !important;
    border-bottom: 3px solid transparent !important;
    background: transparent !important;
    border-radius: 0 !important;
    transition: color 0.15s !important;
    letter-spacing: 0.1px !important;
}
[data-baseweb="tab"]:hover { color: var(--navy) !important; }
[aria-selected="true"][data-baseweb="tab"] {
    color: var(--accent) !important;
    border-bottom: 3px solid var(--accent) !important;
    background: transparent !important;
}
[data-baseweb="tab-panel"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 14px 14px !important;
    padding: 1.75rem 1.75rem !important;
}

/* ── KPI cards ── */
.kpi-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 1rem; margin-bottom: 1.75rem;
}
.kpi-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.35rem 1.2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.055);
    text-align: center; position: relative; overflow: hidden;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
    border-radius: 14px 14px 0 0;
}
.kpi-card.total::before  { background: linear-gradient(90deg, #3B82F6, #60a5fa); }
.kpi-card.high::before   { background: linear-gradient(90deg, #DC2626, #f87171); }
.kpi-card.medium::before { background: linear-gradient(90deg, #D97706, #fbbf24); }
.kpi-card.low::before    { background: linear-gradient(90deg, #16A34A, #4ade80); }
.kpi-label {
    font-size: 0.71rem; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.55rem;
}
.kpi-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.5rem; line-height: 1; margin-bottom: 0.35rem;
}
.kpi-card.total  .kpi-value { color: #3B82F6; }
.kpi-card.high   .kpi-value { color: #DC2626; }
.kpi-card.medium .kpi-value { color: #D97706; }
.kpi-card.low    .kpi-value { color: #16A34A; }
.kpi-sub { font-size: 0.73rem; color: var(--muted); font-weight: 500; }

/* ── Section headings inside tabs ── */
.tab-section-heading {
    font-size: 0.75rem; font-weight: 700; color: var(--navy);
    text-transform: uppercase; letter-spacing: 0.8px;
    margin: 0 0 1rem; padding-bottom: 0.55rem;
    border-bottom: 1.5px solid var(--border);
    display: flex; align-items: center; gap: 0.4rem;
}

/* ── Filter panel inside dashboard tab ── */
.filter-panel {
    background: #F8FAFC; border: 1px solid var(--border);
    border-radius: 12px; padding: 1.1rem 1.4rem;
    margin-bottom: 1.5rem;
}
.filter-panel-title {
    font-size: 0.72rem; font-weight: 700; color: var(--navy);
    text-transform: uppercase; letter-spacing: 0.7px; margin-bottom: 0.85rem;
}

/* ── Risk badges ── */
.badge {
    display: inline-block; padding: 0.24rem 0.65rem;
    border-radius: 20px; font-size: 0.73rem; font-weight: 700;
    letter-spacing: 0.5px; text-transform: uppercase;
}
.badge-high   { background: var(--red-bg);   color: var(--red);   border: 1px solid #FECACA; }
.badge-medium { background: var(--amber-bg); color: var(--amber); border: 1px solid #FDE68A; }
.badge-low    { background: var(--green-bg); color: var(--green); border: 1px solid #BBF7D0; }

/* ── Risk rows ── */
.risk-row {
    display: flex; align-items: center; gap: 1rem;
    padding: 0.88rem 1.1rem; border-radius: 10px;
    margin-bottom: 0.55rem; border: 1px solid transparent;
    transition: box-shadow 0.15s;
}
.risk-row:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.risk-row.high   { background: var(--red-bg);   border-color: #FECACA; }
.risk-row.medium { background: var(--amber-bg); border-color: #FDE68A; }
.risk-row.low    { background: var(--green-bg); border-color: #BBF7D0; }
.risk-icon   { font-size: 1.05rem; flex-shrink: 0; }
.risk-detail { flex: 1; min-width: 0; }
.risk-name   { font-weight: 600; font-size: 0.88rem; color: var(--text); margin-bottom: 0.12rem; }
.risk-clause { font-size: 0.76rem; color: var(--muted); }

/* ── Executive summary premium card ── */
.summary-premium {
    background: linear-gradient(160deg, #f8faff 0%, #eef4ff 100%);
    border: 1px solid #CBD5E1; border-left: 5px solid var(--accent);
    border-radius: 14px; padding: 2rem 2.25rem;
    font-size: 0.92rem; line-height: 1.9; color: var(--text);
    white-space: pre-wrap; font-family: 'Inter', sans-serif;
    box-shadow: 0 4px 20px rgba(59,130,246,0.07);
}
.summary-header {
    display: flex; align-items: center; gap: 0.6rem;
    margin-bottom: 1.25rem; padding-bottom: 0.9rem;
    border-bottom: 1px solid #DDE6F5;
}
.summary-header-icon { font-size: 1.5rem; }
.summary-header-text { font-family: 'DM Serif Display', serif; font-size: 1.15rem; color: var(--navy); }

/* ── Clause table wrapper ── */
.clause-table-wrap {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.stDataFrame { border-radius: 0 !important; }

/* ── Evaluation tab ── */
.eval-score-card {
    background: linear-gradient(135deg, var(--navy) 0%, #1e4a7a 100%);
    border-radius: 16px; padding: 2rem 2.25rem;
    text-align: center; color: #fff; margin-bottom: 1.5rem;
    box-shadow: 0 6px 24px rgba(15,30,54,0.18);
    position: relative; overflow: hidden;
}
.eval-score-card::before {
    content: ''; position: absolute; top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(59,130,246,0.2) 0%, transparent 65%);
    border-radius: 50%;
}
.eval-score-label {
    font-size: 0.75rem; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; color: rgba(255,255,255,0.6); margin-bottom: 0.4rem;
}
.eval-score-value {
    font-family: 'DM Serif Display', serif; font-size: 4rem;
    line-height: 1; color: #fff; position: relative; z-index: 1;
}
.eval-score-sub { font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-top: 0.4rem; }
.eval-metric-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.1rem 1.2rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.eval-metric-card.pass { border-top: 3px solid #16A34A; }
.eval-metric-card.fail { border-top: 3px solid #DC2626; }
.eval-metric-card.warn { border-top: 3px solid #D97706; }
.eval-metric-name {
    font-size: 0.78rem; font-weight: 700; color: var(--navy);
    margin-bottom: 0.3rem; letter-spacing: 0.1px;
}
.eval-metric-status {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px;
}
.eval-metric-status.pass { color: #16A34A; }
.eval-metric-status.fail { color: #DC2626; }
.eval-metric-status.warn { color: #D97706; }
.eval-metric-score { font-size: 1.5rem; font-weight: 700; color: var(--navy); line-height: 1; margin-top: 0.35rem; }

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 4rem 2rem;
    background: var(--card); border: 2px dashed var(--border);
    border-radius: 16px;
}
.empty-state-icon { font-size: 3.2rem; margin-bottom: 0.9rem; }
.empty-state-title { font-size: 1.05rem; font-weight: 600; color: #475569; margin-bottom: 0.45rem; }
.empty-state-sub { font-size: 0.87rem; color: var(--muted); max-width: 380px; margin: 0 auto; line-height: 1.6; }

/* ── Footer ── */
.footer {
    text-align: center; color: var(--muted); font-size: 0.78rem;
    padding: 2rem 0 1rem; border-top: 1px solid var(--border); margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="logo-mark">⚖️</div>
    <div class="header-text">
        <p class="header-title">AI Contract Risk Analyzer</p>
        <p class="header-subtitle">
            Upload a contract to automatically extract clauses, detect risk provisions,
            and generate an executive summary — powered by NLP.
        </p>
    </div>
    <div class="header-badge">Enterprise Edition</div>
</div>
""", unsafe_allow_html=True)


# ─── Helper functions (all unchanged from v2) ─────────────────────────────────
def extract_text_from_upload(uploaded_file) -> str:
    """Extract plain text from uploaded .txt or .pdf file."""
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    elif name.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                st.error("PDF parsing library not found. Install: `pip install pdfplumber`")
                return ""
    return ""


def call_api(text: str) -> dict | None:
    """POST contract text to the FastAPI backend and return the result dict."""
    try:
        resp = requests.post(
            f"{API_BASE}/risk-analyze",
            json={"text": text, "risk_level": None, "top_n": None},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "⚠️ **Cannot connect to the backend API.**  \n"
            "Make sure `python3 app.py` is running at `http://127.0.0.1:8080`."
        )
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None


def severity_icon(level: str) -> str:
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(level.upper(), "⚪")

def severity_class(level: str) -> str:
    return {"HIGH": "high", "MEDIUM": "medium", "LOW": "low"}.get(level.upper(), "low")

def badge_html(level: str) -> str:
    return f'<span class="badge badge-{severity_class(level)}">{level}</span>'

def kpi_card(css_class: str, label: str, value: int, sub: str) -> str:
    return f"""
    <div class="kpi-card {css_class}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""


# ─── Input Section ────────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)
    st.markdown('<div class="input-panel-title">📂 Contract Input</div>', unsafe_allow_html=True)

    col_upload, col_div, col_paste = st.columns([10, 1, 10], gap="small")

    with col_upload:
        st.markdown('<div class="input-label">Upload a file (PDF or TXT)</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            label="Upload PDF or TXT", type=["pdf", "txt"],
            label_visibility="collapsed", help="Supported: PDF, TXT",
        )
        if uploaded_file:
            st.success(f"✅ **{uploaded_file.name}** ready ({uploaded_file.size:,} bytes)")

    with col_div:
        st.markdown('<div class="col-divider">OR</div>', unsafe_allow_html=True)

    with col_paste:
        st.markdown('<div class="input-label">Paste contract text directly</div>', unsafe_allow_html=True)
        pasted_text = st.text_area(
            label="Paste contract text", label_visibility="collapsed",
            placeholder="Paste the full contract text here…", height=175,
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ─── Analyze Button ───────────────────────────────────────────────────────────
st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([3, 2, 3])
with btn_col:
    analyze_clicked = st.button("🔍 Analyze Contract", use_container_width=True)


# ─── Analysis trigger — stores result in session_state ────────────────────────
if analyze_clicked:
    contract_text = ""
    if uploaded_file:
        uploaded_file.seek(0)
        contract_text = extract_text_from_upload(uploaded_file)
    elif pasted_text.strip():
        contract_text = pasted_text.strip()

    if not contract_text:
        st.warning("⚠️ Please upload a file or paste contract text before analyzing.")
        st.stop()

    with st.spinner("Analyzing contract — extracting clauses, detecting risks, generating summary…"):
        result = call_api(contract_text)

    if result is None:
        st.stop()

    # Persist result — filter widget reruns won't trigger a new API call
    st.session_state["analysis_result"] = result


# ─── Results rendering (from session_state) ───────────────────────────────────
if "analysis_result" in st.session_state:
    result = st.session_state["analysis_result"]
    

    # ── Parse raw result (unchanged logic from v2) ────────────────────────────
    clauses = result.get("clauses", [])
    all_risks: list[dict] = []
    for clause in clauses:
        for risk in clause.get("risks", []):
            all_risks.append({
                "risk":         risk.get("risk", ""),
                "level":        risk.get("level", "").upper(),
                "clause_title": clause.get("clause_title", ""),
                "clause_type":  clause.get("clause_type", ""),
                "clause_num":   clause.get("clause_number", ""),
            })

    summary         = result.get("summary", "No summary generated.")
    high_risks      = [r for r in all_risks if r["level"] == "HIGH"]
    medium_risks    = [r for r in all_risks if r["level"] == "MEDIUM"]
    low_risks       = [r for r in all_risks if r["level"] == "LOW"]
    clause_types_available = sorted({c.get("clause_type", "Unknown") for c in clauses})

    # ── Success banner ────────────────────────────────────────────────────────
    st.success(
        f"✅ Analysis complete — **{len(clauses)} clauses** detected, "
        f"**{len(all_risks)} risks** identified."
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # TABBED RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    tab_dashboard, tab_clauses, tab_risks, tab_summary, tab_eval = st.tabs([
        "📊 Dashboard",
        "📄 Clause Analysis",
        "⚠️ Risk Analysis",
        "📝 Executive Summary",
        "📈 Evaluation",
    ])

    # ───────────────────────────────────────────────────────────────────────────
    # TAB 1 — DASHBOARD
    # ───────────────────────────────────────────────────────────────────────────
    with tab_dashboard:

        # KPI Cards
        cards_html = (
            '<div class="kpi-grid">'
            + kpi_card("total",  "📋 Total Clauses",  len(clauses),      "Detected clause groups")
            + kpi_card("high",   "🔴 High Risks",     len(high_risks),   "Critical provisions")
            + kpi_card("medium", "🟡 Medium Risks",   len(medium_risks), "Notable provisions")
            + kpi_card("low",    "🟢 Low Risks",      len(low_risks),    "Minor provisions")
            + "</div>"
        )
        st.markdown(cards_html, unsafe_allow_html=True)

        # Charts
        CHART_HEIGHT = 400
        chart_col1, chart_col2 = st.columns(2, gap="large")

        # ── Bar chart: Clause Distribution ────────────────────────────────────
        with chart_col1:
            if clauses:
                type_counts = (
                    pd.Series([c.get("clause_type", "Unknown") for c in clauses])
                    .value_counts()
                    .rename_axis("Clause Type")
                    .reset_index(name="Count")
                    .sort_values(by=["Count", "Clause Type"], ascending=[False, True])
                    .reset_index(drop=True)
                )
                type_counts.columns = ["Clause Type", "Count"]
                type_counts["Description"] = type_counts["Clause Type"].map(
                    lambda t: CLAUSE_DESCRIPTIONS.get(t, "General contract provision.")
                )
                colors_bar = [
                    f"rgba(42,92,153,{0.6 + 0.4 * (i / max(len(type_counts) - 1, 1))})"
                    for i in range(len(type_counts))
                ]
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=type_counts["Clause Type"],
                    y=type_counts["Count"],
                    text=type_counts["Count"],
                    textposition="outside",
                    textfont=dict(size=13, color="#1A202C", family="Inter"),
                    marker=dict(color=colors_bar, line=dict(width=0), cornerradius=6),
                    customdata=list(zip(type_counts["Count"], type_counts["Description"])),
                    hovertemplate=(
                        "<b>%{x}</b><br>Count: <b>%{customdata[0]}</b><br><br>"
                        "<i>%{customdata[1]}</i><extra></extra>"
                    ),
                ))
                fig_bar.update_layout(
                    title=dict(
                        text="<b>Clause Distribution Analysis</b>",
                        font=dict(size=16, color="#0F1E36", family="DM Serif Display"),
                        x=0.5, xanchor="center",
                    ),
                    paper_bgcolor="white", plot_bgcolor="#F7F9FC", showlegend=False,
                    xaxis=dict(
                        title=dict(text="Clause Categories", font=dict(size=12, color="#64748B")),
                        tickangle=-30, tickfont=dict(size=11, family="Inter"),
                        gridcolor="#E2E8F0", linecolor="#E2E8F0",
                    ),
                    yaxis=dict(
                        title=dict(text="Number of Clauses", font=dict(size=12, color="#64748B")),
                        tickfont=dict(size=11, family="Inter"),
                        gridcolor="#E2E8F0", linecolor="#E2E8F0", zeroline=False,
                    ),
                    margin=dict(l=50, r=30, t=60, b=100),
                    height=CHART_HEIGHT,
                    hoverlabel=dict(bgcolor="white", bordercolor="#E2E8F0", font=dict(size=13, family="Inter")),
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.markdown('<div class="empty-state"><div class="empty-state-icon">📊</div><div class="empty-state-title">No clause data</div><div class="empty-state-sub">No clauses were detected to chart.</div></div>', unsafe_allow_html=True)

        # ── Donut chart: Risk Severity Distribution ────────────────────────────
        with chart_col2:
            if all_risks:
                risk_by_level   = {"HIGH": high_risks, "MEDIUM": medium_risks, "LOW": low_risks}
                sev_labels      = [k for k, v in risk_by_level.items() if v]
                sev_counts      = [len(v) for v in risk_by_level.values() if v]
                sev_colors      = {"HIGH": "#DC2626", "MEDIUM": "#D97706", "LOW": "#16A34A"}
                sev_colors_list = [sev_colors[l] for l in sev_labels]
                top_risk_texts  = []
                for lvl in sev_labels:
                    top5 = [r["risk"] for r in risk_by_level[lvl][:5]]
                    top_risk_texts.append("<br>".join(f"• {r}" for r in top5))

                fig_donut = go.Figure(data=[go.Pie(
                    labels=sev_labels, values=sev_counts, hole=0.55,
                    marker=dict(colors=sev_colors_list, line=dict(color="white", width=3)),
                    textinfo="label+percent",
                    textfont=dict(size=13, family="Inter"),
                    customdata=top_risk_texts,
                    hovertemplate=(
                        "<b>Risk Level: %{label}</b><br>Count: <b>%{value}</b><br>"
                        "Share: %{percent}<br><br><b>Top Risks:</b><br>%{customdata}<extra></extra>"
                    ),
                    pull=[0.04 if l == "HIGH" else 0 for l in sev_labels],
                )])
                fig_donut.update_layout(
                    title=dict(
                        text="<b>Risk Severity Distribution</b>",
                        font=dict(size=16, color="#0F1E36", family="DM Serif Display"),
                        x=0.5, xanchor="center",
                    ),
                    paper_bgcolor="white", showlegend=True,
                    legend=dict(orientation="v", x=0.82, y=0.5, font=dict(size=12, family="Inter"), itemsizing="constant"),
                    annotations=[dict(
                        text=f"<b style='font-size:22px'>{len(all_risks)}</b><br>"
                             "<span style='font-size:11px;color:#64748B'>total risks</span>",
                        x=0.5, y=0.5, font=dict(size=13, color="#0F1E36"), showarrow=False,
                    )],
                    margin=dict(l=20, r=110, t=60, b=30),
                    height=CHART_HEIGHT,
                    hoverlabel=dict(bgcolor="white", bordercolor="#E2E8F0", font=dict(size=13, family="Inter"), align="left"),
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.markdown('<div class="empty-state"><div class="empty-state-icon">⚠️</div><div class="empty-state-title">No risk data</div><div class="empty-state-sub">No risks were detected to chart.</div></div>', unsafe_allow_html=True)

        # ── Executive overview strip ───────────────────────────────────────────
        st.markdown('<div class="tab-section-heading">📋 Executive Overview</div>', unsafe_allow_html=True)
        overall = "HIGH" if high_risks else ("MEDIUM" if medium_risks else "LOW")
        overall_color = {"HIGH": "#DC2626", "MEDIUM": "#D97706", "LOW": "#16A34A"}[overall]
        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);border-radius:12px;
                    padding:1.1rem 1.4rem;display:flex;align-items:center;gap:1.2rem;
                    box-shadow:0 1px 6px rgba(0,0,0,0.04);">
            <div style="flex:1;">
                <div style="font-size:0.75rem;font-weight:700;color:var(--muted);
                            text-transform:uppercase;letter-spacing:0.7px;margin-bottom:0.2rem;">
                    Overall Risk Level
                </div>
                <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:{overall_color};">
                    {overall}
                </div>
            </div>
            <div style="text-align:right;font-size:0.82rem;color:var(--muted);line-height:1.8;">
                <strong style="color:var(--text);">{len(clauses)}</strong> clauses &nbsp;·&nbsp;
                <strong style="color:#DC2626;">{len(high_risks)}</strong> high &nbsp;·&nbsp;
                <strong style="color:#D97706;">{len(medium_risks)}</strong> medium &nbsp;·&nbsp;
                <strong style="color:#16A34A;">{len(low_risks)}</strong> low risks
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Report ID
        if result.get("id"):
            st.markdown(
                f'<div style="text-align:right;font-size:0.75rem;color:#94A3B8;margin-top:0.75rem;">'
                f'Report ID: <code>{result["id"]}</code></div>',
                unsafe_allow_html=True,
            )

    # ───────────────────────────────────────────────────────────────────────────
    # TAB 2 — CLAUSE ANALYSIS
    # ───────────────────────────────────────────────────────────────────────────
    with tab_clauses:
        st.markdown('<div class="tab-section-heading">📄 Detected Clauses</div>', unsafe_allow_html=True)

        # ── Filter (inside clause analysis tab) ────────────────────────────────
        st.markdown('<div class="tab-section-heading">🎛️ Clause Filters</div>', unsafe_allow_html=True)
        with st.container():
            cf1, _, _ = st.columns([1.2, 1, 1.6], gap="large")
            with cf1:
                clause_type_filter = st.selectbox(
                    "Clause Type",
                    options=["All Clauses"] + clause_types_available,
                    index=0,
                )

        # ── Apply filter (unchanged logic from v2) ─────────────────────────────
        filtered_clauses = clauses[:]
        if clause_type_filter != "All Clauses":
            filtered_clauses = [c for c in filtered_clauses if c.get("clause_type") == clause_type_filter]

        if filtered_clauses:
            # Stats row
            c1, c2, c3 = st.columns(3, gap="medium")
            c1.metric("Total Clauses", len(clauses))
            c2.metric("Filtered Clauses", len(filtered_clauses))
            c3.metric("Clause Types", len(clause_types_available))
            st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

            clause_df = pd.DataFrame([
                {
                    "Clause No": c.get("clause_number", ""),
                    "Title":     c.get("clause_title", ""),
                    "Type":      c.get("clause_type", ""),
                }
                for c in filtered_clauses
            ])
            st.dataframe(
                clause_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Clause No": st.column_config.NumberColumn(width="small"),
                    "Title":     st.column_config.TextColumn(width="medium"),
                    "Type":      st.column_config.TextColumn(width="medium"),
                },
            )
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">📄</div>
                <div class="empty-state-title">No clauses match the selected filter</div>
                <div class="empty-state-sub">Try changing the Clause Type filter above.</div>
            </div>
            """, unsafe_allow_html=True)

    # ───────────────────────────────────────────────────────────────────────────
    # TAB 3 — RISK ANALYSIS
    # ───────────────────────────────────────────────────────────────────────────
    with tab_risks:
        st.markdown('<div class="tab-section-heading">⚠️ Risk Analysis</div>', unsafe_allow_html=True)

        # ── Filters (inside risk analysis tab) ──────────────────────────────────
        st.markdown('<div class="tab-section-heading">🎛️ Risk Filters</div>', unsafe_allow_html=True)
        with st.container():
            rf1, rf2, _ = st.columns([1.2, 1, 1.6], gap="large")
            with rf1:
                severity_filter = st.selectbox(
                    "Risk Severity",
                    options=["All Risks", "High Risk Only", "Medium Risk Only", "Low Risk Only"],
                    index=0,
                )
            with rf2:
                top_n_filter = st.number_input(
                    "Number of Top Risks to Display",
                    min_value=1,
                    max_value=max(len(all_risks), 1),
                    value=min(5, max(len(all_risks), 1)),
                    step=1,
                )

        # ── Apply filters (unchanged logic from v2) ─────────────────────────────
        level_map     = {"All Risks": None, "High Risk Only": "HIGH", "Medium Risk Only": "MEDIUM", "Low Risk Only": "LOW"}
        chosen_level  = level_map[severity_filter]
        filtered_risks = all_risks[:]
        if chosen_level:
            filtered_risks = [r for r in filtered_risks if r["level"] == chosen_level]
        filtered_risks = filtered_risks[:int(top_n_filter)]

        if filtered_risks:
            # Summary strip
            r1, r2, r3 = st.columns(3, gap="medium")
            r1.metric("🔴 High",   len([r for r in filtered_risks if r["level"] == "HIGH"]))
            r2.metric("🟡 Medium", len([r for r in filtered_risks if r["level"] == "MEDIUM"]))
            r3.metric("🟢 Low",    len([r for r in filtered_risks if r["level"] == "LOW"]))
            st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

            for risk in filtered_risks:
                lvl  = risk["level"]
                cls  = severity_class(lvl)
                icon = severity_icon(lvl)
                st.markdown(f"""
                <div class="risk-row {cls}">
                    <span class="risk-icon">{icon}</span>
                    <div class="risk-detail">
                        <div class="risk-name">{risk['risk']}</div>
                        <div class="risk-clause">Clause {risk['clause_num']}: {risk['clause_title']}</div>
                    </div>
                    {badge_html(lvl)}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">🛡️</div>
                <div class="empty-state-title">No risks match the selected filter</div>
                <div class="empty-state-sub">Try adjusting the Risk Severity or Top N filters above.</div>
            </div>
            """, unsafe_allow_html=True)

    # ───────────────────────────────────────────────────────────────────────────
    # TAB 4 — EXECUTIVE SUMMARY
    # ───────────────────────────────────────────────────────────────────────────
    with tab_summary:
        st.markdown('<div class="tab-section-heading">📝 Executive Summary</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="summary-premium">
            <div class="summary-header">
                <span class="summary-header-icon">📋</span>
                <span class="summary-header-text">Contract Risk Assessment Report</span>
            </div>{summary}</div>
        """, unsafe_allow_html=True)

        # Download summary as text
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download Summary",
            data=summary,
            file_name="contract_summary.txt",
            mime="text/plain",
            use_container_width=False,
        )

    # ───────────────────────────────────────────────────────────────────────────
    # TAB 5 — EVALUATION
    # ───────────────────────────────────────────────────────────────────────────
    with tab_eval:
        st.markdown('<div class="tab-section-heading">📈 Evaluation</div>', unsafe_allow_html=True)

        evaluation = result.get("evaluation", {})

        if not evaluation:
            st.warning("No evaluation data returned from backend.")
        else:
            overall = evaluation.get("overall_quality_score", 0)

            st.markdown(f'''
<div class="eval-score-card">
<div class="eval-score-label">Overall Pipeline Score</div>
<div class="eval-score-value">{overall}%</div>
<div class="eval-score-sub">
Clause:  {evaluation.get("clause_extraction",{}).get("percentage",0)}% |
Risk: {evaluation.get("risk_detection",{}).get("percentage",0)}% |
Summary: {evaluation.get("summary_quality",{}).get("percentage",0)}%
</div>
</div>
''', unsafe_allow_html=True)

            with st.expander("Clause Evaluation", expanded=False):
                clause_names={
                    "valid_json":"JSON Validation",
                    "correct_clause_count":"Clause Count Accuracy",
                    "correct_classification":"Classification Accuracy",
                    "no_hallucination":"Hallucination Check",
                    "prompt_compliance":"Prompt Compliance"
                }

                for item in evaluation["clause_extraction"]["criteria"]:
                    pct=0 if item["max_score"]==0 else item["score"]/item["max_score"]
                    st.markdown(f"**{clause_names.get(item['criterion'],item['criterion'])}**")
                    st.progress(pct)
                    a,b,c=st.columns([1,1,5])
                    a.metric("Score",f"{item['score']}/{item['max_score']}")
                    b.metric("Status","PASS" if item["passed"] else "FAIL")
                    c.info(item["detail"])

            if "summary_quality" in evaluation:
                with st.expander("Summary Evaluation", expanded=False):
                    names={
                        "length_compliance":"Length Compliance",
                        "no_hallucination":"Hallucination Check",
                        "risk_coverage":"Risk Coverage",
                        "clause_coverage":"Clause Coverage",
                        "format_compliance":"Format Compliance"
                    }
                    for item in evaluation["summary_quality"]["criteria"]:
                        pct=0 if item["max_score"]==0 else item["score"]/item["max_score"]
                        st.markdown(f"**{names.get(item['criterion'],item['criterion'])}**")
                        st.progress(pct)
                        a,b,c=st.columns([1,1,5])
                        a.metric("Score",f"{item['score']}/{item['max_score']}")
                        b.metric("Status","PASS" if item["passed"] else "FAIL")
                        c.info(item["detail"])

            with st.expander("Combined Scores", expanded=False):
                c1,c2,c3=st.columns(3)
                c1.metric("Clause Score",f"{evaluation['clause_extraction']['percentage']}%")
                c2.metric("Risk Score ",f"{evaluation['risk_detection']['percentage']}%")
                if "summary_quality" in evaluation:
                    c3.metric("Summary Score",f"{evaluation['summary_quality']['percentage']}%")
                else:
                    c3.metric("Summary Score","N/A")


else:
    # ── Empty state (no analysis run yet) ─────────────────────────────────────
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">⚖️</div>
        <div class="empty-state-title">Ready to analyze your contract</div>
        <div class="empty-state-sub">
            Upload a PDF or TXT file, or paste contract text above,
            then click <strong>Analyze Contract</strong> to get started.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    AI Contract Risk Analyzer &nbsp;·&nbsp; Powered by FastAPI + Streamlit &nbsp;·&nbsp;
    <span style="color:#3B82F6;">Enterprise Edition</span>
</div>
""", unsafe_allow_html=True)
