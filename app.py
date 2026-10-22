"""Van Subscription — Underwriting Consistency.

Premium instrument-panel demo for a London van-subscription founder.
Real ML underwriting on Hugging Face loan data; synthetic applicant framing.

Framed as vetting consistency, never credit scoring.
Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import random
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from models import Applicant
from ml_engine import vet, describe_decision, normalise_trade_category
from synthetic import generate_applicant, generate_paraphrases, apply_paraphrase
from consistency import (
    run_consistency_test,
    batch_consistency_test,
    aggregate_flip_rate,
    find_flip_prone_applicant,
)
from fit_finder import evaluate_fit
from affordability import (
    assess_applicant_deal,
    applicant_van_cost,
    applicant_total_van_cost,
    powertrain_for_affordability,
)

st.set_page_config(
    page_title="Van Sub — Underwriting Consistency",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)
TOKENS_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;500;700&family=IBM+Plex+Mono:wght@400;500;700&display=swap');

:root {
  --bg: #0F0E0C;
  --surface: #171512;
  --surface-high: #201D19;
  --border: rgba(243,238,228,0.09);
  --text: #F3EEE4;
  --text-muted: #A39C8F;
  --accent: #FF5B2E;
  --accent-hover: #FF7448;
  --accent-soft: rgba(255,91,46,0.14);
  --on-accent: #0F0E0C;
  --signal: #C8F169;
  --warning: #F5B83D;
  --danger: #E5484D;

  /* Semantic panel tints.
     These are DARK-tinted surfaces, not light ones. The app theme is dark
     (--bg #0F0E0C, --text #F3EEE4), so any panel that hard-codes a light
     pastel background will render its text at roughly 1:1 contrast and
     become unreadable. Each panel below is a low-saturation tint of the
     semantic hue over --surface-high, which keeps the hue readable while
     holding every foreground above the WCAG AA 4.5:1 threshold.

     Panel        background   foreground   fg/bg   text/bg   muted/bg
     ok           #313221      #C8F169      10.10    11.29     4.79
     warn         #352C1D      #F5B83D       7.72    11.87     5.04
     danger       #392421      #FF5A5F       4.75    12.54     5.32
     info         #2A2C32      #7AA2FF       5.61    12.07     5.12
  */
  --panel-ok-bg: #313221;
  --panel-ok-fg: #C8F169;
  --panel-ok-border: #5E6B37;
  --panel-warn-bg: #352C1D;
  --panel-warn-fg: #F5B83D;
  --panel-warn-border: #6F5627;
  --panel-danger-bg: #392421;
  --panel-danger-fg: #FF5A5F;
  --panel-danger-border: #743434;
  --panel-info-bg: #2A2C32;
  --panel-info-fg: #7AA2FF;
  --panel-info-border: #424F70;

  --radius: 6px;
  --transition: 220ms cubic-bezier(0.23, 1, 0.32, 1);
  --spring: 350ms cubic-bezier(0.34, 1.56, 0.64, 1);
}

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'IBM Plex Mono', monospace !important;
}

[data-testid="stAppViewContainer"] > .main {
  background: transparent !important;
  padding-top: 0 !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stDecoration"] { visibility: hidden; }
[data-testid="stStatusWidget"] { visibility: hidden; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

[data-testid="stAppViewContainer"]::before {
  content: "";
  position: fixed;
  inset: 0;
  background-image: radial-gradient(circle at 50% 50%, rgba(255,91,46,0.03) 0%, transparent 40%);
  pointer-events: none;
  z-index: 0;
}

::selection { background: var(--accent-soft); color: var(--text); }
:focus-visible { outline: none; box-shadow: 0 0 0 2px var(--accent), 0 0 0 4px var(--bg); }

.stButton > button,
.stDownloadButton > button {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

.container {
  max-width: 1180px;
  margin: 0 auto;
  padding: 40px 24px 80px 24px;
  position: relative;
  z-index: 1;
}

.section { margin-bottom: 64px; }

.section-head {
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 24px;
}

.section-head .kicker {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 500;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.16em;
  color: var(--accent);
  padding-bottom: 2px;
  border-bottom: 1px solid var(--border);
}

.section-head h2 {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 1.5rem;
  letter-spacing: -0.02em;
  color: var(--text);
  margin: 0;
  line-height: 1.2;
}

.surface {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  position: relative;
  overflow: hidden;
}

.surface-alt { background: var(--surface-high); }

/* A semantic status panel (recommendation banner, affordability verdict).
   The background and foreground are supplied by the caller via the
   --panel-bg / --panel-fg custom properties, so a panel can never
   accidentally pair a light background with the dark theme's light text. */
.panel {
  background: var(--panel-bg, var(--surface-high));
  border: 1px solid var(--panel-border, var(--border));
  border-radius: var(--radius);
  padding: 24px;
  position: relative;
  overflow: hidden;
}

.panel-title {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 800;
  font-size: 1.25rem;
  color: var(--text);
  margin-top: 6px;
}

.panel-figure {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 2rem;
  line-height: 1.2;
  letter-spacing: -0.02em;
  color: var(--panel-fg, var(--accent));
}

.panel-figure-sm { font-size: 1.6rem; }

.panel-note {
  color: var(--text-muted);
  font-size: 0.85rem;
}

/* A chip sitting on a semantic panel. It uses the panel's own foreground
   colour so it stays legible against the tinted background. */
.panel-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  font-family: 'Space Grotesk', sans-serif;
  white-space: nowrap;
  background: var(--panel-bg, var(--surface-high));
  color: var(--panel-fg, var(--text));
  border: 1px solid var(--panel-border, var(--border));
}

h1, h2, h3, .display {
  font-family: 'Space Grotesk', sans-serif;
  color: var(--text);
}

.display-1 {
  font-size: 3rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  line-height: 1.1;
  margin: 0 0 8px 0;
}

.display-2 {
  font-size: 1.75rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.2;
  margin: 0 0 12px 0;
}

.lead {
  font-size: 1rem;
  line-height: 1.6;
  color: var(--text-muted);
  margin: 0 0 24px 0;
}

.mono { font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; }
.text-muted { color: var(--text-muted); }

.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  font-family: 'Space Grotesk', sans-serif;
  border: 1px solid transparent;
  white-space: nowrap;
}

.chip-approve { background: rgba(200,241,105,0.12); color: var(--signal); border-color: rgba(200,241,105,0.25); }
.chip-refer   { background: rgba(245,184,61,0.12); color: var(--warning); border-color: rgba(245,184,61,0.25); }
.chip-decline { background: rgba(229,72,77,0.12); color: var(--danger); border-color: rgba(229,72,77,0.25); }
.chip-accent  { background: var(--accent); color: var(--on-accent); }

.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 10px 20px;
  border-radius: var(--radius);
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 500;
  font-size: 0.9rem;
  border: 1px solid var(--border);
  background: var(--surface-high);
  color: var(--text);
  cursor: pointer;
  transition: all var(--transition);
  text-decoration: none;
}

.btn:hover { border-color: rgba(243,238,228,0.35); background: var(--surface); transform: translateY(-1px); }
.btn:active { transform: translateY(0); }

.btn-primary { background: var(--accent); color: var(--on-accent); border-color: var(--accent); }
.btn-primary:hover { background: var(--accent-hover); border-color: var(--accent-hover); transform: translateY(-1px); box-shadow: 0 0 20px var(--accent-soft); }
.btn-ghost { background: transparent; border-color: transparent; color: var(--text-muted); }
.btn-ghost:hover { color: var(--text); background: var(--surface-high); }

.metric-group {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 16px;
  margin-bottom: 32px;
}

.metric {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
}

.metric .label {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--text-muted);
  margin-bottom: 8px;
  font-weight: 500;
}

.metric .value {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 1.75rem;
  letter-spacing: -0.02em;
  color: var(--text);
  line-height: 1.2;
}
.metric .value.accent { color: var(--accent); }
.metric .value.signal { color: var(--signal); }

/* The same value styles, usable OUTSIDE a .metric block. Previously these
   rules were scoped to ".metric .value", so any markup using class="value"
   outside a metric tile (the recommendation banner, the affordability
   verdict) silently fell back to the inherited body colour. On a tinted
   panel that made the figure unreadable. */
.value {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 1.75rem;
  letter-spacing: -0.02em;
  color: var(--text);
  line-height: 1.2;
}
.value.accent { color: var(--accent); }
.value.signal { color: var(--signal); }

.metric .delta {
  font-size: 0.75rem;
  font-family: 'IBM Plex Mono', monospace;
  color: var(--text-muted);
  margin-top: 4px;
}

.reason-pill {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: 500;
  color: var(--text);
  border: 1px solid var(--border);
  background: var(--surface-high);
  margin: 4px 6px 4px 0;
}
.reason-detail {
  font-size: 0.8rem;
  color: var(--text-muted);
  margin: 2px 0 8px 0;
  padding-left: 14px;
  border-left: 1px solid var(--border);
}

.para-row {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 10px;
  background: var(--surface);
  transition: border-color var(--transition), background var(--transition);
}
.para-row:hover { border-color: rgba(243,238,228,0.25); }
.para-row.base { background: var(--accent-soft); border-color: rgba(255,91,46,0.35); }

.para-tag {
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-muted);
  padding: 2px 8px;
  border-radius: 4px;
  background: var(--surface-high);
  border: 1px solid var(--border);
  white-space: nowrap;
}
.para-row.base .para-tag { color: var(--accent); border-color: var(--accent); }

.para-trade {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 500;
  font-size: 0.95rem;
  color: var(--text);
  flex: 1;
}
.para-decision { display: flex; flex-direction: column; align-items: flex-end; gap: 4px; text-align: right; }
.para-score {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.75rem;
  color: var(--text-muted);
}

.decision-wire { position: relative; height: 120px; margin: 32px 0; }
.wire-track {
  position: absolute; left: 0; right: 0; top: 56px; height: 2px;
  background: var(--border); border-radius: 1px;
}
.wire-tick {
  position: absolute; top: 50px; width: 10px; height: 10px;
  border: 1px solid var(--border); border-radius: 50%; background: var(--surface);
  cursor: pointer; transition: all var(--transition); z-index: 2;
}
.wire-tick:hover { border-color: var(--text); transform: scale(1.2); z-index: 3; }
.wire-tick-approve { left: 0%; }
.wire-tick-refer   { left: 50%; }
.wire-tick-decline { left: 100%; }
.wire-label {
  position: absolute; top: 68px; font-family: 'Space Grotesk', sans-serif;
  font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--text-muted);
}
.wire-label-approve { left: 0%; transform: translateX(-50%); }
.wire-label-refer   { left: 50%; transform: translateX(-50%); }
.wire-label-decline { left: 100%; transform: translateX(-50%); }
.wire-marker {
  position: absolute; top: 36px; width: 26px; height: 26px;
  background: var(--accent); border: 2px solid var(--bg); border-radius: 50%;
  box-shadow: 0 0 24px var(--accent-soft);
  transition: left var(--spring), top var(--spring); z-index: 4; cursor: pointer;
}
.wire-marker.pos-approve { left: 0%; top: 36px; transform: translateX(-50%); }
.wire-marker.pos-refer   { left: 50%; top: 36px; transform: translateX(-50%); }
.wire-marker.pos-decline { left: 100%; top: 36px; transform: translateX(-50%); }
.wire-glow {
  position: absolute; top: 50%; left: 50%; width: 200px; height: 60px;
  background: radial-gradient(ellipse at center, var(--accent-soft) 0%, transparent 70%);
  border-radius: 50%; transform: translate(-50%, -50%); opacity: 0;
  transition: opacity var(--spring), transform var(--spring); z-index: 1;
}

.nav {
  position: fixed; top: 0; left: 0; right: 0; height: 64px;
  border-bottom: 1px solid var(--border);
  background: rgba(15,14,12,0.8); backdrop-filter: blur(8px);
  z-index: 100; display: flex; align-items: center;
  justify-content: space-between; padding: 0 24px;
}
.nav-mark {
  font-family: 'Space Grotesk', sans-serif; font-weight: 700;
  font-size: 1.25rem; letter-spacing: -0.02em; color: var(--text);
  text-decoration: none; display: flex; align-items: center; gap: 8px;
}
.nav-dot { width: 8px; height: 8px; background: var(--accent); border-radius: 50%; }
.nav-links { display: flex; gap: 24px; list-style: none; margin: 0; padding: 0; }
.nav-links a {
  font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; font-weight: 500;
  color: var(--text-muted); text-decoration: none; padding: 8px 0;
  border-bottom: 2px solid transparent; transition: color var(--transition);
}
.nav-links a:hover, .nav-links a.active {
  color: var(--text); border-bottom-color: var(--accent);
}

.footer {
  border-top: 1px solid var(--border); padding: 32px 24px;
  color: var(--text-muted); font-size: 0.75rem;
  font-family: 'IBM Plex Mono', monospace; text-align: center; margin-top: 64px;
}

@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}
.reveal { animation: fadeInUp var(--spring) both; }
.reveal-1 { animation-delay: 60ms; }
.reveal-2 { animation-delay: 120ms; }
.reveal-3 { animation-delay: 180ms; }
.reveal-4 { animation-delay: 240ms; }

.stButton > button {
  width: 100%; border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
  background: var(--surface-high) !important;
  color: var(--text) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 500 !important;
  transition: all var(--transition) !important;
}
.stButton > button:hover {
  border-color: rgba(243,238,228,0.35) !important;
  background: var(--surface) !important;
  transform: translateY(-1px) !important;
}
.stDownloadButton > button {
  width: 100%; border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
  background: var(--surface-high) !important;
  color: var(--text) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 500 !important;
}

/* Normalise st.metric to the app's metric cards. Streamlit's default
   renders a 36px IBM Plex Mono value, which clashes with the 1.75rem
   Space Grotesk values used by the custom .metric cards. */
[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
}

[data-testid="stMetricLabel"] {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 0.7rem !important;
  font-weight: 500 !important;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--text-muted) !important;
}

[data-testid="stMetricValue"] {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 1.75rem !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em;
  color: var(--text) !important;
  line-height: 1.2;
}

[data-testid="stMetricDelta"] {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.75rem !important;
  color: var(--text-muted) !important;
}

/* Nav button overrides — flat, text-like, active underline.
   Streamlit puts a `st-key-<key>` class on each button's element container,
   so we can target the nav row precisely regardless of how many wrapper
   divs Streamlit nests around it. */
div[class*="st-key-nav_"] div[data-testid="stButton"] > button {
  border: none !important;
  background: transparent !important;
  color: var(--text-muted) !important;
  padding: 10px 0 !important;
  border-radius: 0 !important;
  border-bottom: 2px solid transparent !important;
  font-size: 0.85rem !important;
  font-weight: 500 !important;
  transition: all var(--transition) !important;
  box-shadow: none !important;
}

div[class*="st-key-nav_"] div[data-testid="stButton"] > button[kind="primary"] {
  color: var(--text) !important;
  border-bottom-color: var(--accent) !important;
}

div[class*="st-key-nav_"] div[data-testid="stButton"] > button:hover {
  color: var(--text) !important;
  background: transparent !important;
  transform: none !important;
  border-color: transparent !important;
  border-bottom-color: var(--accent) !important;
}

"""

st.markdown(f"<style>{TOKENS_CSS}</style>", unsafe_allow_html=True)

if "applicant" not in st.session_state:
    st.session_state.applicant = generate_applicant(42)
if "report" not in st.session_state:
    st.session_state.report = None
if "batch_agg" not in st.session_state:
    st.session_state.batch_agg = None
if "batch_reports" not in st.session_state:
    st.session_state.batch_reports = None
if "active_page" not in st.session_state:
    st.session_state.active_page = "Dashboard"


def decision_chip(decision: str) -> str:
    css = {"approve": "chip-approve", "refer": "chip-refer", "decline": "chip-decline"}[decision]
    label = {"approve": "Approve", "refer": "Refer", "decline": "Decline"}[decision]
    return f'<span class="chip {css}">{label}</span>'

def reason_pill(reason: str) -> str:
    return f'<span class="reason-pill">{reason.replace("_", " ").title()}</span>'

def reason_detail(detail: str) -> str:
    return f'<div class="reason-detail">{detail}</div>'

def to_csv_df(report) -> pd.DataFrame:
    rows = []
    for p in report.paraphrases:
        d = p.decision
        rows.append({
            "wording": p.label,
            "trade_as_written": p.trade_description,
            "decision": d.decision,
            "vetting_score": d.score,
            "confidence": d.confidence,
            "flipped_vs_base": p.flipped,
            "reason_codes": "; ".join(d.reasons),
            "reason_details": " | ".join(d.reason_details),
        })
    return pd.DataFrame(rows)

def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# --- Navigation ---------------------------------------------------------------
_PAGES = [
    "Dashboard",
    "Consistency",
    "EV vs Diesel",
    "Scenario",
    "Export",
]

# A URL like ?page=Consistency should deep-link into the app. We track the
# LAST URL value we applied, rather than reading query_params blindly on
# every rerun. Reading it blindly would stomp on the session_state value the
# nav buttons set, so the page would snap straight back to Dashboard.
_url_page = st.query_params.get("page", None)
if isinstance(_url_page, list):
    _url_page = _url_page[0] if _url_page else None
if _url_page and _url_page in _PAGES:
    if st.session_state.get("_url_page_applied") != _url_page:
        st.session_state.active_page = _url_page
        st.session_state._url_page_applied = _url_page

if st.session_state.get("active_page") not in _PAGES:
    st.session_state.active_page = "Dashboard"

page = st.session_state.active_page

_nav_cols = st.columns(len(_PAGES))
for _i, _key in enumerate(_PAGES):
    _is_active = page == _key
    with _nav_cols[_i]:
        if st.button(
            _key,
            key=f"nav_{_key}",
            type="primary" if _is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_page = _key
            st.query_params["page"] = _key
            st.rerun()

st.markdown('<div class="container">', unsafe_allow_html=True)

# ===========================================================================
# DASHBOARD
# ===========================================================================

if page == "Dashboard":
    st.markdown(
        '<div class="section-head reveal reveal-1">'
        '<span class="kicker">Headline</span>'
        '<h2>The one number that matters</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="surface reveal reveal-2">'
        '<p class="lead">'
        'Run a batch of synthetic SME applicants through the ML underwriter under several '
        'different trade wordings — same financial facts, different descriptions. The flip rate '
        'is the share of paraphrases where the decision changes.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    c_run, c_note = st.columns([1, 2])
    with c_run:
        if st.button("Run batch sweep — 50 applicants", key="btn_batch", type="primary"):
            apps = [generate_applicant(seed=s) for s in range(50)]
            reports = batch_consistency_test(apps, n_paraphrases=5, seed=7)
            st.session_state.batch_reports = reports
            st.session_state.batch_agg = aggregate_flip_rate(reports)
            st.rerun()
    with c_note:
        st.caption(
            "Each applicant is vetted 5× with different trade wordings. Financial facts stay identical. "
            "Model: Gradient-boosted classifier trained on AnguloM/loan_data (Hugging Face, 9,578 real applications), "
            "plus an explicit trade-risk-tier overlay."
        )

    if st.session_state.batch_agg:
        agg = st.session_state.batch_agg
        st.markdown(
            '<div class="metric-group reveal reveal-3">'
            f'<div class="metric">'
            f'<span class="label">Overall flip rate</span>'
            f'<span class="value accent">{agg["overall_flip_rate"]:.0%}</span>'
            f'<span class="delta">{agg["flipped_count"]} of {agg["total_paraphrases"]} paraphrases flipped</span>'
            f'</div>'
            f'<div class="metric">'
            f'<span class="label">Reason stability</span>'
            f'<span class="value">{agg["avg_reason_stability"]:.0%}</span>'
            f'<span class="delta">share with identical reasons</span>'
            f'</div>'
            f'<div class="metric">'
            f'<span class="label">Applicants tested</span>'
            f'<span class="value">{agg["applicants_tested"]}</span>'
            f'<span class="delta">synthetic SME applicants</span>'
            f'</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-head reveal reveal-4">'
            '<span class="kicker">Readout</span>'
            '<h2>What this means</h2>'
            '</div>',
            unsafe_allow_html=True,
        )

        if agg["overall_flip_rate"] >= 0.10:
            st.markdown(
                f'<div class="surface surface-alt reveal reveal-4">'
                f'<p style="margin:0; color: var(--danger); font-weight: 600;">'
                f'⚠ {agg["overall_flip_rate"]:.0%} of paraphrases flip the decision.</p>'
                f'<p style="margin:8px 0 0 0; color: var(--text-muted); font-size: 0.9rem;">'
                f'Material inconsistency — tighten trade-category keywords or add a second '
                f'reviewer for edge cases before go-live.'
                f'</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif agg["overall_flip_rate"] >= 0.04:
            st.markdown(
                f'<div class="surface surface-alt reveal reveal-4">'
                f'<p style="margin:0; color: var(--warning); font-weight: 600;">'
                f'⚡ {agg["overall_flip_rate"]:.0%} flip rate — moderate inconsistency.</p>'
                f'<p style="margin:8px 0 0 0; color: var(--text-muted); font-size: 0.9rem;">'
                f'Concentrated in borderline cases. Review applicants whose decisions span '
                f'approve ↔ decline.'
                f'</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="surface surface-alt reveal reveal-4">'
                f'<p style="margin:0; color: var(--signal); font-weight: 600;">'
                f'✓ {agg["overall_flip_rate"]:.0%} flip rate — low inconsistency.</p>'
                f'<p style="margin:8px 0 0 0; color: var(--text-muted); font-size: 0.9rem;">'
                f'Keep monitoring as real applicant data comes in.'
                f'</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="reveal reveal-4" style="margin-top:32px;">', unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        dist = agg["decision_distribution"]
        if dist:
            df_dist = pd.DataFrame(
                [{"decision": k.title(), "count": v} for k, v in dist.items()]
            )
            fig = px.pie(
                df_dist, names="decision", values="count",
                title="Base decisions across 50 applicants",
                color_discrete_map={"Approve": "#C8F169", "Refer": "#F5B83D", "Decline": "#E5484D"},
            )
            fig.update_traces(textposition="inside", textinfo="label+percent")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="IBM Plex Mono", color="#F3EEE4", size=11),
                title=dict(font=dict(family="Space Grotesk", size=13, color="#F3EEE4")),
                legend=dict(font=dict(color="#A39C8F")),
                margin=dict(t=40, b=20, l=20, r=20),
            )
            d1.plotly_chart(fig, use_container_width=True)

        flip_hist = [r.flip_rate for r in (st.session_state.batch_reports or [])]
        if flip_hist:
            df_flip = pd.DataFrame({"flip_rate": flip_hist})
            fig2 = px.histogram(
                df_flip, x="flip_rate", nbins=10,
                title="Flip-rate distribution per applicant",
            )
            fig2.update_xaxes(tickformat=".0%", title="Flip rate", gridcolor="rgba(243,238,228,0.1)")
            fig2.update_yaxes(title="Applicants", gridcolor="rgba(243,238,228,0.1)")
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="IBM Plex Mono", color="#F3EEE4", size=11),
                title=dict(font=dict(family="Space Grotesk", size=13, color="#F3EEE4")),
                margin=dict(t=40, b=20, l=20, r=20),
            )
            d2.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        worst = max(st.session_state.batch_reports, key=lambda r: r.flip_rate)
        st.markdown(
            f'<div class="section-head reveal reveal-4" style="margin-top:48px;">'
            f'<span class="kicker">Worst case</span>'
            f'<h2>{worst.applicant.business_name}</h2>'
            f'</div>',
            unsafe_allow_html=True,
        )
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Applicant", worst.applicant.business_name)
        w2.metric("Trade", worst.applicant.trade.title())
        w3.metric("Flip rate", f"{worst.flip_rate:.0%}")
        w4.metric("Base decision", worst.base_decision.title())
        st.caption(
            f"Base reasons: {', '.join(worst.base_reasons) or 'none'} · "
            f"reason stability: {worst.reason_stability_rate:.0%}"
        )

# ===========================================================================
# CONSISTENCY TESTER
# ===========================================================================

elif page == "Consistency":
    st.markdown(
        '<div class="section-head reveal reveal-1">'
        '<span class="kicker">Consistency Tester</span>'
        '<h2>One business, many wordings</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="surface reveal reveal-2">'
        '<p class="lead">'
        'The same SME facts are re-worded — only the trade description changes. Each wording runs '
        'through the ML underwriter. The Decision Wire shows where each paraphrase lands; any '
        'movement is inconsistency.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    app = st.session_state.applicant
    # Generated applicants start with an empty trade_category; normalise it
    # on read so the caption always shows the real tier rather than blank.
    _app_tier = app.trade_category or normalise_trade_category(app.trade)
    c_run, c_meta = st.columns([1, 2])
    with c_run:
        n_para = st.slider("Paraphrases", 3, 8, 5, key="slider_para")
        if st.button("Run consistency test", key="btn_consistency", type="primary"):
            st.session_state.report = run_consistency_test(
                app, n_paraphrases=n_para, seed=7
            )
            st.rerun()
    with c_meta:
        st.caption(
            f"{app.business_name} · {app.trade} · "
            f"category: {_app_tier.title()} risk"
        )

    if st.session_state.report:
        rep = st.session_state.report
        base = rep.paraphrases[0]
        pos_map = {"approve": "pos-approve", "refer": "pos-refer", "decline": "pos-decline"}
        base_pos = pos_map[base.decision.decision]

        st.markdown('<div class="reveal reveal-3">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="decision-wire">'
            f'<div class="wire-track"></div>'
            f'<div class="wire-tick wire-tick-approve" title="Approve"></div>'
            f'<div class="wire-tick wire-tick-refer" title="Refer"></div>'
            f'<div class="wire-tick wire-tick-decline" title="Decline"></div>'
            f'<div class="wire-label wire-label-approve">Approve</div>'
            f'<div class="wire-label wire-label-refer">Refer</div>'
            f'<div class="wire-label wire-label-decline">Decline</div>'
            f'<div class="wire-marker {base_pos}"></div>'
            f'<div class="wire-glow"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        b1, b2, b3, b4 = st.columns(4)
        b1.markdown(decision_chip(base.decision.decision), unsafe_allow_html=True)
        b2.metric("Vetting score", f"{base.decision.score:.1f}")
        b3.metric("Confidence", f"{base.decision.confidence:.0%}")
        b4.metric("Threshold", f"{base.decision.threshold:.2f}")

        st.markdown("**Why — base wording**")
        st.markdown(" ".join(reason_pill(r) for r in base.decision.reasons) or "<i>none</i>", unsafe_allow_html=True)
        for detail in base.decision.reason_details:
            st.markdown(reason_detail(detail), unsafe_allow_html=True)

        # --- Affordability verdict -------------------------------------
        # The credit decision alone is not the answer to "should we lease
        # this van?". A business can be perfectly creditworthy and still
        # not be able to carry the running cost. This panel joins the
        # decision to the affordability of the deal, using the same cost
        # model the credit engine scored against.
        st.markdown('<div class="reveal reveal-3" style="margin-top:24px;">', unsafe_allow_html=True)
        st.markdown(
            '<h3 style="font-family: Space Grotesk; font-size: 0.85rem; text-transform: uppercase; '
            'letter-spacing: 0.1em; color: var(--accent); margin-bottom: 12px;">'
            'Can this deal work?</h3>',
            unsafe_allow_html=True,
        )

        fit_for_verdict = evaluate_fit(rep.applicant)
        deal = assess_applicant_deal(
            rep.applicant, base.decision.decision, fit_for_verdict.recommendation
        )

        deal_chip = {
            "workable": ("chip-approve", "WORKABLE"),
            "tight": ("chip-refer", "TIGHT"),
            "not_workable": ("chip-decline", "NOT WORKABLE"),
        }[deal.verdict]

        d1, d2, d3 = st.columns(3)
        d1.markdown(
            f'<span class="chip {deal_chip[0]}">{deal_chip[1]}</span>',
            unsafe_allow_html=True,
        )
        d2.metric("Total / month", f"£{deal.monthly_cost_total:,.0f}")
        d3.metric("Of turnover", f"{deal.commitment_pct:.0f}%")

        st.markdown(deal.headline)
        for reason in deal.reasons:
            st.markdown(reason_detail(reason), unsafe_allow_html=True)

        if deal.actions:
            st.markdown("**What to do next**")
            for action in deal.actions:
                st.markdown(f"- {action}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="reveal reveal-3" style="margin-top:24px;">', unsafe_allow_html=True)
        st.markdown('<h3 style="font-family: Space Grotesk; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent); margin-bottom: 12px;">Paraphrase results</h3>', unsafe_allow_html=True)

        s1, s2, s3 = st.columns(3)
        s1.metric("Flips vs base", rep.flip_count)
        s2.metric("Flip rate", f"{rep.flip_rate:.0%}")
        s3.metric("Reason stability", f"{rep.reason_stability_rate:.0%}")

        if rep.flip_count > 0:
            st.markdown(
                f'<div class="surface surface-alt" style="margin-top:12px;">'
                f'<p style="margin:0; color: var(--danger); font-weight: 600;">'
                f'⚠ {rep.flip_count} of {rep.total_paraphrases} paraphrases flipped the decision.</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="surface surface-alt" style="margin-top:12px;">'
                f'<p style="margin:0; color: var(--signal); font-weight: 600;">'
                f'✓ All paraphrases agreed — consistent for this business.</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        for p in rep.paraphrases:
            d = p.decision
            is_base = p.label == "original"
            base_css = " para-row base" if is_base else ""
            chip_html = decision_chip(d.decision)
            reasons_html = " ".join(reason_pill(r) for r in d.reasons) or "<i>none</i>"
            st.markdown(
                f'<div class="para-row{base_css}">'
                f'<span class="para-tag">{p.label}</span>'
                f'<div class="para-trade">{p.trade_description}</div>'
                f'<div class="para-decision">'
                f'{chip_html}'
                f'<span class="para-score">{d.score:.1f}</span>'
                f'</div>'
                f'<div style="display:flex; gap:6px; flex-wrap:wrap;">{reasons_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

        df_one = to_csv_df(rep)
        st.download_button(
            "Download consistency report (CSV)",
            data=csv_bytes(df_one),
            file_name=f"consistency_{rep.applicant.business_name.replace(' ','_')}.csv",
            mime="text/csv",
            key="dl_consistency",
        )

# ===========================================================================
# EV vs DIESEL
# ===========================================================================

elif page == "EV vs Diesel":
    st.markdown(
        '<div class="section-head reveal reveal-1">'
        '<span class="kicker">Fit Finder</span>'
        '<h2>EV vs diesel</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="surface reveal reveal-2">'
        '<p class="lead">'
        'Given mileage, charging access, payload and budget, the fit finder scores electric vs '
        'diesel suitability and estimates monthly running costs for a London van subscription.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    app = st.session_state.applicant
    fit = evaluate_fit(app)

    rec_label = {"ev": "Electric van recommended", "diesel": "Diesel van recommended", "hybrid": "Either could work"}
    # Semantic panel tokens. These are DARK tints: the app is a dark theme, so
    # a light background would render the near-white body text at ~1:1
    # contrast and make the banner unreadable.
    rec_panel = {
        "ev": ("--panel-ok-bg", "--panel-ok-fg", "--panel-ok-border"),
        "diesel": ("--panel-warn-bg", "--panel-warn-fg", "--panel-warn-border"),
        "hybrid": ("--panel-info-bg", "--panel-info-fg", "--panel-info-border"),
    }[fit.recommendation]
    panel_bg, panel_fg, panel_border = rec_panel

    st.markdown(
        f'<div class="surface panel reveal reveal-3" '
        f'style="--panel-bg: var({panel_bg}); --panel-fg: var({panel_fg}); '
        f'--panel-border: var({panel_border});">'
        f'<div style="display:flex; justify-content: space-between; align-items: center;">'
        f'<div>'
        f'<span class="panel-chip">{rec_label[fit.recommendation].upper()}</span>'
        f'<div class="panel-title">{fit.confidence:.0%} confidence</div>'
        f'</div>'
        f'<div style="text-align: right;">'
        f'<span class="panel-figure">EV {fit.ev_score:.0f} / 100</span>'
        f'<div class="panel-note">diesel {fit.diesel_score:.0f} / 100</div>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="metric-group reveal reveal-3" style="margin-top:24px;">', unsafe_allow_html=True)
    ev_over = app.monthly_budget_gbp > 0 and fit.est_monthly_ev_gbp > app.monthly_budget_gbp
    ds_over = app.monthly_budget_gbp > 0 and fit.est_monthly_diesel_gbp > app.monthly_budget_gbp
    st.markdown(
        f'<div class="metric">'
        f'<span class="label">EV / month</span>'
        f'<span class="value accent">£{fit.est_monthly_ev_gbp:,.0f}</span>'
        f'<span class="delta">{">budget" if ev_over else ""}</span>'
        f'</div>'
        f'<div class="metric">'
        f'<span class="label">Diesel / month</span>'
        f'<span class="value">£{fit.est_monthly_diesel_gbp:,.0f}</span>'
        f'<span class="delta">{">budget" if ds_over else ""}</span>'
        f'</div>'
        f'<div class="metric">'
        f'<span class="label">Applicant budget</span>'
        f'<span class="value">£{app.monthly_budget_gbp:,}</span>' if app.monthly_budget_gbp
        else '<span class="value">not set</span>'
        f'</div>'
        f'<div class="metric">'
        f'<span class="label">Cheaper option</span>'
        f'<span class="value signal">{"EV" if fit.est_monthly_ev_gbp < fit.est_monthly_diesel_gbp else "Diesel"}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="reveal reveal-3" style="margin-top:32px;">', unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown('**Electric van**')
        st.progress(int(fit.ev_score))
        st.caption(f"{fit.ev_score:.0f} / 100 suitability")
    with g2:
        st.markdown('**Diesel van**')
        st.progress(int(fit.diesel_score))
        st.caption(f"{fit.diesel_score:.0f} / 100 suitability")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<h3 style="font-family: Space Grotesk; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent); margin: 32px 0 12px 0;">Why</h3>', unsafe_allow_html=True)
    for r in fit.reasons:
        tag = "EV" if r.supports_ev and not r.supports_diesel else ("Diesel" if r.supports_diesel and not r.supports_ev else "Both")
        tag_css = "chip-approve" if tag == "EV" else ("chip-refer" if tag == "Diesel" else "")
        st.markdown(
            f'<div style="display:flex; align-items: center; gap: 8px; margin-bottom: 8px;">'
            f'<span class="chip {tag_css}" style="font-size: 0.7rem;">{tag}</span>'
            f'<div style="flex: 1;">'
            f'<div style="font-weight: 700; color: var(--text); font-size: 0.9rem;">{r.label}</div>'
            f'<div style="color: var(--text-muted); font-size: 0.8rem;">{r.detail}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.info(fit.note)

    # ------------------------------------------------------------------
    # Affordability verdict
    # ------------------------------------------------------------------
    # The credit decision and the vehicle fit are two halves of the same
    # question. This panel joins them: can the business actually carry the
    # cost of the vehicle we are about to recommend? It uses the SAME cost
    # model that the credit engine scored against, so the two can never
    # disagree.
    st.divider()
    st.markdown(
        '<h3 style="font-family: Space Grotesk; font-size: 0.85rem; text-transform: uppercase; '
        'letter-spacing: 0.1em; color: var(--accent); margin: 32px 0 12px 0;">'
        'Can this deal work?</h3>',
        unsafe_allow_html=True,
    )

    # The credit decision for this applicant.
    credit = vet(app)

    # The powertrain the affordability verdict prices. When the fit finder
    # says "either could work", the cheaper option is used - that is the
    # best case actually available to this applicant.
    priced_powertrain = powertrain_for_affordability(app, fit.recommendation)
    priced_cost = applicant_van_cost(app, priced_powertrain).total

    verdict = assess_applicant_deal(app, credit.decision, fit.recommendation)

    verdict_panel = {
        "workable": ("--panel-ok-bg", "--panel-ok-fg", "--panel-ok-border"),
        "tight": ("--panel-warn-bg", "--panel-warn-fg", "--panel-warn-border"),
        "not_workable": ("--panel-danger-bg", "--panel-danger-fg", "--panel-danger-border"),
    }[verdict.verdict]
    panel_bg, panel_fg, panel_border = verdict_panel

    st.markdown(
        f'<div class="surface panel reveal reveal-3" '
        f'style="--panel-bg: var({panel_bg}); --panel-fg: var({panel_fg}); '
        f'--panel-border: var({panel_border});">'
        f'<div style="display:flex; justify-content: space-between; align-items: center;">'
        f'<div>'
        f'<span class="panel-chip">{verdict.verdict.replace("_", " ").upper()}</span>'
        f'<div class="panel-title" style="font-size: 1.15rem;">{verdict.headline}</div>'
        f'</div>'
        f'<div style="text-align: right;">'
        f'<span class="panel-figure panel-figure-sm">£{verdict.monthly_cost_total:,.0f}</span>'
        f'<div class="panel-note" style="font-size: 0.8rem;">per month, all vans</div>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="metric-group reveal reveal-3" style="margin-top:16px;">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="metric">'
        f'<span class="label">Cost / van ({priced_powertrain})</span>'
        f'<span class="value">£{priced_cost:,.0f}</span>'
        f'<span class="delta">per month</span>'
        f'</div>'
        f'<div class="metric">'
        f'<span class="label">Total commitment</span>'
        f'<span class="value">£{verdict.monthly_cost_total:,.0f}</span>'
        f'<span class="delta">{verdict.commitment_pct:.0f}% of turnover</span>'
        f'</div>'
        f'<div class="metric">'
        f'<span class="label">Monthly turnover</span>'
        f'<span class="value">£{verdict.monthly_turnover:,.0f}</span>'
        f'<span class="delta">{"under 15% guideline" if verdict.commitment_pct <= 15 else "over 15% guideline"}</span>'
        f'</div>'
        f'<div class="metric">'
        f'<span class="label">Credit decision</span>'
        f'<span class="value">{credit.decision.title()}</span>'
        f'<span class="delta">{credit.confidence:.0%} confidence</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if verdict.budget > 0:
        if verdict.within_budget:
            st.success(
                f"£{verdict.budget_gap:,.0f}/month of headroom under the stated budget "
                f"of £{verdict.budget:,.0f}."
            )
        else:
            st.error(
                f"Over the stated budget by £{abs(verdict.budget_gap):,.0f}/month "
                f"(budget £{verdict.budget:,.0f}, cost £{verdict.monthly_cost_total:,.0f})."
            )

    st.markdown("**Why**")
    for reason in verdict.reasons:
        st.markdown(reason_detail(reason), unsafe_allow_html=True)

    if verdict.actions:
        st.markdown("**What to do next**")
        for action in verdict.actions:
            st.markdown(f"- {action}")

    st.divider()
    st.markdown('<h3 style="font-family: Space Grotesk; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent); margin: 32px 0 12px 0;">Applicant usage profile</h3>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Daily mileage", f"{app.daily_mileage} miles")
    p2.metric("Urban share", f"{app.urban_mileage_pct}%")
    p3.metric("Charging access", "Yes" if (app.home_charging or app.depot_charging) else "No")
    p4.metric("Payload / size", f"{app.payload_kg} kg ({app.van_size})")
    st.caption(
        "Cost model: lease + energy/fuel + maintenance + road tax, 22 days/month. "
        "EVs avoid London congestion charge & ULEZ where charging access exists. Demo planning figures, not quotes."
    )

# ===========================================================================
# SCENARIO BUILDER
# ===========================================================================

elif page == "Scenario":
    st.markdown(
        '<div class="section-head reveal reveal-1">'
        '<span class="kicker">Scenario Builder</span>'
        '<h2>Tune the facts, watch consistency live</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="surface reveal reveal-2">'
        '<p class="lead">'
        'Manually adjust an applicant’s facts, then re-run the consistency test to see how sensitive '
        'the decision is. Stress-test edge cases before they hit production underwriting.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    app = st.session_state.applicant
    e1, e2, e3 = st.columns(3)
    trade_text = e1.text_input("Trade description", value=app.trade, key="sc_trade")
    years = e2.number_input("Years in business", 0.0, 50.0, float(app.years_in_business), 0.5, key="sc_years")
    turnover = e3.number_input("Annual turnover (£)", 0, 5_000_000, int(app.annual_turnover_gbp), step=5000, key="sc_turnover")

    e4, e5, e6 = st.columns(3)
    fleet = e4.number_input("Existing fleet size", 0, 100, int(app.fleet_size), key="sc_fleet")
    requested = e5.number_input("Vans requested", 1, 50, int(app.requested_vans), key="sc_requested")
    adverse = e6.checkbox("Adverse credit (CCJ/default)", value=bool(app.adverse_credit), key="sc_adverse")

    e7, e8, e9 = st.columns(3)
    _use_labels = {"tradesman_transport": "Tradesman transport", "delivery": "Delivery", "mixed": "Mixed"}
    primary_use = e7.selectbox(
        "Primary use",
        ["tradesman_transport", "delivery", "mixed"],
        index=["tradesman_transport", "delivery", "mixed"].index(app.primary_use),
        format_func=lambda k: _use_labels[k],
        key="sc_use",
    )
    urban = e8.slider("Urban mileage %", 0, 100, int(app.urban_mileage_pct), key="sc_urban")
    daily = e9.number_input("Daily mileage (miles)", 0, 500, int(app.daily_mileage), key="sc_daily")

    e10, e11, e12 = st.columns(3)
    payload = e10.number_input("Payload (kg)", 0, 3000, int(app.payload_kg), key="sc_payload")
    van_size = e11.selectbox("Van size", ["small", "medium", "large"], index=["small", "medium", "large"].index(app.van_size), key="sc_size")
    budget = e12.number_input("Monthly budget (£)", 0, 5000, int(app.monthly_budget_gbp), step=50, key="sc_budget")

    e13, e14 = st.columns(2)
    home_charge = e13.checkbox("Home charging", value=bool(app.home_charging), key="sc_home")
    depot_charge = e14.checkbox("Depot charging", value=bool(app.depot_charging), key="sc_depot")

    if st.button("Apply changes & re-test", key="btn_scenario", type="primary"):
        edited = Applicant(
            business_name=app.business_name,
            trade=trade_text,
            trade_category=normalise_trade_category(trade_text),
            years_in_business=years,
            employee_count=app.employee_count,
            annual_turnover_gbp=float(turnover),
            adverse_credit=adverse,
            fleet_size=int(fleet),
            requested_vans=int(requested),
            primary_use=primary_use,
            urban_mileage_pct=int(urban),
            daily_mileage=int(daily),
            home_charging=home_charge,
            depot_charging=depot_charge,
            payload_kg=int(payload),
            van_size=van_size,
            monthly_budget_gbp=int(budget),
        )
        st.session_state.applicant = edited
        st.session_state.report = run_consistency_test(edited, n_paraphrases=5, seed=7)
        st.rerun()

    if st.session_state.report:
        rep = st.session_state.report
        base = rep.paraphrases[0]
        pos_map = {"approve": "pos-approve", "refer": "pos-refer", "decline": "pos-decline"}
        base_pos = pos_map[base.decision.decision]

        st.divider()
        st.markdown(
            '<div class="section-head reveal reveal-3">'
            '<span class="kicker">Result</span>'
            '<h2>After your changes</h2>'
            '</div>',
            unsafe_allow_html=True,
        )
        r1, r2, r3, r4 = st.columns(4)
        r1.markdown(decision_chip(base.decision.decision), unsafe_allow_html=True)
        r2.metric("Vetting score", f"{base.decision.score:.1f}")
        r3.metric("Flip rate", f"{rep.flip_rate:.0%}")
        r4.metric("Reason stability", f"{rep.reason_stability_rate:.0%}")

        if rep.flip_count > 0:
            st.markdown(
                f'<div class="surface surface-alt reveal reveal-3" style="margin-top:12px;">'
                f'<p style="margin:0; color: var(--danger); font-weight: 600;">'
                f'⚠ {rep.flip_count} of {rep.total_paraphrases} paraphrases flip ({rep.flip_rate:.0%}). '
                f'This scenario is sensitive to wording.'
                f'</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="surface surface-alt reveal reveal-3" style="margin-top:12px;">'
                f'<p style="margin:0; color: var(--signal); font-weight: 600;">'
                f'✓ Decision holds steady across all paraphrases — consistent for this scenario.</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="reveal reveal-3" style="margin-top:16px;">', unsafe_allow_html=True)
        st.markdown("**Reason codes (base wording):**")
        st.markdown(" ".join(reason_pill(r) for r in base.decision.reasons) or "<i>none</i>", unsafe_allow_html=True)
        for detail in base.decision.reason_details:
            st.markdown(reason_detail(detail), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# EXPORT REPORT
# ===========================================================================

elif page == "Export":
    st.markdown(
        '<div class="section-head reveal reveal-1">'
        '<span class="kicker">Export</span>'
        '<h2>Compliance audit trail</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="surface reveal reveal-2">'
        '<p class="lead">'
        'Download the latest consistency test as CSV for compliance review, board packs, or '
        'handing to the engineering team to fix the underwriter.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    rep = st.session_state.report
    if rep is None:
        st.info("Run a consistency test first (Consistency or Scenario).")
    else:
        df_one = to_csv_df(rep)
        st.markdown("### Single-applicant consistency report")
        st.caption(
            f"Applicant: {rep.applicant.business_name} · base decision: {rep.base_decision.title()} "
            f"· flip rate: {rep.flip_rate:.0%} · reason stability: {rep.reason_stability_rate:.0%}"
        )
        st.dataframe(df_one, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV",
            data=csv_bytes(df_one),
            file_name=f"consistency_{rep.applicant.business_name.replace(' ','_')}_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            key="dl_export_single",
        )

        if st.session_state.batch_reports:
            st.divider()
            st.markdown("### Batch sweep — all 50 applicants")
            rows = []
            for r in st.session_state.batch_reports:
                rows.append({
                    "applicant": r.applicant.business_name,
                    "trade": r.applicant.trade,
                    "category": r.applicant.trade_category
                    or normalise_trade_category(r.applicant.trade),
                    "base_decision": r.base_decision,
                    "base_reasons": "; ".join(r.base_reasons),
                    "flip_count": r.flip_count,
                    "flip_rate": r.flip_rate,
                    "reason_stability": r.reason_stability_rate,
                })
            df_batch = pd.DataFrame(rows)
            st.dataframe(df_batch.sort_values("flip_rate", ascending=False), use_container_width=True, hide_index=True)
            st.download_button(
                "Download batch CSV",
                data=csv_bytes(df_batch),
                file_name=f"batch_consistency_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
                key="dl_export_batch",
            )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer">'
    'Van-subscription underwriting demo · synthetic SME applicants · vetting consistency, not credit scoring · '
    'Outcomes: approve / refer / decline · ML model: gradient-boosted classifier on AnguloM/loan_data (Hugging Face)'
    '</div>',
    unsafe_allow_html=True,
)

