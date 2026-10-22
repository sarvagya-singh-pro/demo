# Van Subscription — Underwriting Consistency Demo

A prototype **vetting consistency tester** and **EV vs diesel van fit finder** for a London van-subscription
business serving tradespeople and SMEs (~500 vans, 1–10 staff, GBP 1.5m seed raised Sept 2026).

The company says it is building an AI underwriting engine to make decisions faster and more consistent.
This demo shows what that engine's *consistency* looks like under the hood — and where it can wobble —
without building a credit model.

## Framing: vetting consistency, not credit scoring

This is **not a credit score**. Outcomes are fleet-subscription vetting decisions:

- **approve** — welcome the business onto the van subscription
- **refer** — send to manual review (edge case)
- **decline** — do not onboard

Every decision is explainable with plain-English reason codes so an underwriter (or a customer) can see
exactly why, and a compliance team can audit it.

## What the demo contains

| Module | Purpose |
|--------|---------|
| `models.py` | Shared dataclasses: `Applicant`, `DecisionResult`, `ConsistencyReport`, `EVVsDieselFit`, `FitReason` |
| `engine.py` | Transparent rules-based vetting engine (stand-in for the AI underwriter) with plain-English reason codes |
| `synthetic.py` | Synthetic SME applicant generator + paraphrase (re-wording) engine |
| `consistency.py` | Runs the same business under N wordings and measures decision flips |
| `fit_finder.py` | EV vs diesel van fit finder with indicative monthly costs (lease, fuel/energy, maintenance, road tax, congestion/ULEZ) |
| `app.py` | Streamlit UI (two tabs: consistency tester + EV vs diesel fit finder, plus a 50-applicant batch sweep) |

## How to run

```bash
# from /Users/sarvagyasingh/Documents/demo
pip3 install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

### Consistency Tester tab
1. Generate (or regenerate) a synthetic SME applicant.
2. Click **Run consistency test on this applicant**.
3. You'll see the base decision plus, for each re-wording of the same business:
   - the trade as written,
   - the decision (approve / refer / decline),
   - vetting score,
   - whether it flipped vs the base,
   - plain-English reason codes.
4. Key metrics: **decision flip rate** and **reason-set stability**. A high flip rate means the same
   business gets different outcomes just because it was described differently — exactly the kind of
   inconsistency an underwriting engine should eliminate before go-live.
5. Click **Batch test 50 applicants** for an aggregate flip rate, decision distribution, and a
   per-applicant flip-rate histogram.

### EV vs Diesel Fit Finder tab
Shows, for the current applicant, whether an electric or diesel van is the better fit, with:
- recommendation (EV / diesel / hybrid) and confidence,
- 0–100 suitability scores,
- plain-English reason codes tagged EV / diesel / both,
- estimated monthly cost for each,
- the applicant's monthly budget for context.

## Example consistency output (illustrative)

```
Business:  Spark London Ltd
Trade:     electrician  →  re-worded as "electrical contractor", "domestic electrician", ...
Decision:  Approve (score 78/60)
Flip rate: 20%  (1 of 5 paraphrases flipped to Refer)
Reason codes: established_business, strong_turnover, clean_credit, low_risk_trade
```

A "flipped" paraphrase usually happens because a synonym (e.g. "electrical contractor" vs "electrician")
maps to a different risk tier in the engine — the very inconsistency this tool is designed to surface.

## Files to show the company

If you're walking into a meeting, bring:
- `streamlit run app.py` — the live, clickable demo
- The Batch Sweep tab — an aggregate "X% of decisions flip under re-wording" headline number
- Plain-English reason codes on every decision — the audit/compliance story

## Notes

- All applicants and figures are synthetic/demo data. Cost assumptions in `fit_finder.py` are
  indicative planning figures, not quotes or FCA-regulated credit advice.
- The engine is intentionally rules-based and fully explainable so it can be replaced by a real ML
  underwriting model later — the consistency test harness stays the same.
