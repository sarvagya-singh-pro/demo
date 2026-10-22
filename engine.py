"""Rule-based underwriting engine (stand-in for the AI underwriter).

Transparent, rules-based vetting so every decision is explainable with
plain-English reason codes. Not a credit score - a fleet-subscription vetting
decision (approve / refer / decline). Deliberately sensitive to how the trade
is worded, which the consistency tester exposes.
"""

from __future__ import annotations

from typing import List, Tuple

from models import Applicant, Decision, DecisionResult
from trade_taxonomy import (
    normalise_trade_category as _taxonomy_normalise_trade_category,
)

# Thresholds ------------------------------------------------------------------

MIN_YEARS_APPROVE = 2.0
MIN_YEARS_REFER = 1.0
MIN_TURNOVER_APPROVE = 40_000
MIN_TURNOVER_REFER = 25_000
MAX_FLEET_SIZE = 20
MAX_REQUESTED_VANS = 10
MIN_STAFF_PER_VAN = 0.5

# Vetting score weights --------------------------------------------------------

WEIGHT_YEARS = 25
WEIGHT_TURNOVER = 25
WEIGHT_STAFF = 15
WEIGHT_FLEET = 15
WEIGHT_CREDIT = 20

# Trade risk tiers ------------------------------------------------------------
# The tier itself is NOT defined here any more - it lives in
# ``trade_taxonomy.py`` so that this rules engine and the ML engine resolve
# the SAME tier for the SAME trade wording.
#
# Previously this module kept its own keyword sets and did naive substring
# matching, which mis-classified trades ("town plumbing" matched the
# high-risk keyword "tow") and let the two engines disagree about the tier
# of the same business.


def normalise_trade_category(trade: str) -> str:
    """Map a free-text trade description to a risk tier.

    Delegates to :func:`trade_taxonomy.normalise_trade_category`, which
    resolves the trade from a canonical vocabulary using whole-word matching.
    The tier is a property of the trade, not of the sentence describing it,
    so all paraphrases of one trade resolve to the same tier.
    """
    return _taxonomy_normalise_trade_category(trade)


def engine_score(applicant: Applicant) -> Tuple[float, List[str], List[str]]:
    """Return (score, reason_codes, reason_details). Score is 0-100."""

    reasons: List[str] = []
    details: List[str] = []
    score = 50.0  # neutral start

    # --- Years in business ---
    years = applicant.years_in_business
    if years >= MIN_YEARS_APPROVE:
        score += WEIGHT_YEARS
        reasons.append("established_business")
        details.append(
            f"Business trading {years:.1f} years (established, >= {MIN_YEARS_APPROVE} yrs)"
        )
    elif years >= MIN_YEARS_REFER:
        score += WEIGHT_YEARS * 0.5
        reasons.append("business_moderately_established")
        details.append(
            f"Business trading {years:.1f} years (between {MIN_YEARS_REFER} and {MIN_YEARS_APPROVE} yrs)"
        )
    else:
        reasons.append("business_too_new")
        details.append(
            f"Business trading less than {MIN_YEARS_REFER} year — limited track record"
        )

    # --- Turnover ---
    turnover = applicant.annual_turnover_gbp
    if turnover >= MIN_TURNOVER_APPROVE:
        score += WEIGHT_TURNOVER
        reasons.append("strong_turnover")
        details.append(
            f"Annual turnover £{turnover:,.0f} (>= £{MIN_TURNOVER_APPROVE:,})"
        )
    elif turnover >= MIN_TURNOVER_REFER:
        score += WEIGHT_TURNOVER * 0.5
        reasons.append("moderate_turnover")
        details.append(
            f"Annual turnover £{turnover:,.0f} (between £{MIN_TURNOVER_REFER:,} and £{MIN_TURNOVER_APPROVE:,})"
        )
    else:
        reasons.append("turnover_too_low")
        details.append(
            f"Annual turnover £{turnover:,.0f} below £{MIN_TURNOVER_REFER:,} threshold"
        )


    # --- Staffing vs requested fleet ---
    staff_per_van = (
        applicant.employee_count / applicant.requested_vans
        if applicant.requested_vans
        else 0
    )
    if staff_per_van >= MIN_STAFF_PER_VAN:
        score += WEIGHT_STAFF
        reasons.append("adequate_staffing")
        details.append(
            f"{applicant.employee_count} staff for {applicant.requested_vans} requested van(s)"
        )
    else:
        reasons.append("insufficient_staffing")
        details.append(
            f"Only {applicant.employee_count} staff for {applicant.requested_vans} requested van(s) "
            f"(ratio {staff_per_van:.2f} staff/van, want >= {MIN_STAFF_PER_VAN})"
        )

    # --- Existing fleet size ---
    if applicant.fleet_size >= MAX_FLEET_SIZE:
        reasons.append("fleet_at_limit")
        details.append(
            f"Already operates {applicant.fleet_size} vans (cap {MAX_FLEET_SIZE})"
        )
        score -= WEIGHT_FLEET
    elif applicant.fleet_size >= MAX_FLEET_SIZE * 0.75:
        reasons.append("fleet_near_limit")
        details.append(
            f"Already operates {applicant.fleet_size} vans, approaching cap of {MAX_FLEET_SIZE}"
        )
        score -= WEIGHT_FLEET * 0.5
    else:
        score += WEIGHT_FLEET * 0.5
        reasons.append("fleet_capacity_headroom")
        details.append(
            f"Operates {applicant.fleet_size} vans, well under the {MAX_FLEET_SIZE} cap"
        )

    # --- Credit ---
    if applicant.adverse_credit:
        reasons.append("adverse_credit")
        details.append(
            "Adverse credit history recorded (CCJ / default / insolvency flag)"
        )
        score -= WEIGHT_CREDIT
    else:
        score += WEIGHT_CREDIT * 0.5
        reasons.append("clean_credit")
        details.append("No adverse credit history recorded")

    # --- Trade risk tier ---
    category = applicant.trade_category or normalise_trade_category(applicant.trade)
    if category == "low":
        score += 10
        reasons.append("low_risk_trade")
        details.append(f"Trade '{applicant.trade}' classed as low risk")
    elif category == "high":
        score -= 15
        reasons.append("high_risk_trade")
        details.append(f"Trade '{applicant.trade}' classed as higher risk")
    else:
        reasons.append("medium_risk_trade")
        details.append(f"Trade '{applicant.trade}' classed as medium risk")

    # Clamp
    score = max(0.0, min(100.0, score))
    return score, reasons, details


def vet(applicant: Applicant) -> DecisionResult:
    """Run the vetting engine on one applicant description.

    Returns a DecisionResult with a plain-English explainable outcome.
    """
    score, reasons, details = engine_score(applicant)
    threshold = 60.0

    if score >= 75:
        decision: Decision = "approve"
        confidence = min(0.95, 0.5 + (score - 75) / 100)
    elif score >= threshold:
        decision = "refer"
        confidence = 0.6
    else:
        decision = "decline"
        confidence = min(0.95, 0.5 + (threshold - score) / 100)

    return DecisionResult(
        decision=decision,
        confidence=round(confidence, 2),
        reasons=reasons,
        reason_details=details,
        score=round(score, 1),
        threshold=threshold,
        applicant=applicant,
    )


def describe_decision(result: DecisionResult) -> str:
    """Plain-English one-line summary of a decision."""
    verbs = {
        "approve": "Approved",
        "refer": "Referred for manual review",
        "decline": "Declined",
    }
    return f"{verbs[result.decision]} — vetting score {result.score}/{result.threshold:.0f} ({result.confidence:.0%} confidence)"
