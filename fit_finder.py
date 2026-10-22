"""EV vs diesel van fit finder.

Recommends electric, diesel, or hybrid van fit for an SME applicant based on
usage patterns (urban mileage, daily mileage, charging access, payload,
budget). Framed as vehicle-fit vetting, not credit scoring.

Output: EVVsDieselFit with recommendation, confidence, 0-100 suitability
scores, plain-English reason codes, and estimated monthly costs.
"""

from __future__ import annotations

from typing import List

from models import Applicant, EVVsDieselFit, FitReason, VanFit
from affordability import van_monthly_cost

# Cost assumptions -------------------------------------------------------------
# These are NOT defined here any more. They live in ``affordability.py`` so
# that the fit finder and the credit engine quote the SAME number for the
# same van. Previously this file had its own copies of the lease, fuel,
# maintenance, tax and charge assumptions, which meant the price the fit
# finder quoted to the user was not the price the credit model was scoring.

def evaluate_fit(applicant: Applicant) -> EVVsDieselFit:
    """Evaluate EV vs diesel fit for an applicant."""
    daily = applicant.daily_mileage
    urban_pct = applicant.urban_mileage_pct
    home_charge = applicant.home_charging
    depot_charge = applicant.depot_charging
    charging_access = home_charge or depot_charge
    payload = applicant.payload_kg
    budget = applicant.monthly_budget_gbp
    primary_use = applicant.primary_use

    reasons: List[FitReason] = []
    ev_score = 50.0
    diesel_score = 50.0
    # --- Daily mileage / range ---
    if daily <= 100:
        ev_score += 20
        reasons.append(
            FitReason(
                code="ev_low_daily_mileage",
                label="Low daily mileage suits EV range",
                detail=f"{daily} miles/day is comfortably within typical EV van range.",
                supports_ev=True,
                supports_diesel=False,
            )
        )
        diesel_score -= 5
    elif daily <= 180:
        ev_score += 5
        diesel_score += 5
        reasons.append(
            FitReason(
                code="moderate_daily_mileage",
                label="Moderate daily mileage",
                detail=f"{daily} miles/day — EV workable with mid-day top-up; diesel has no range concern.",
                supports_ev=True,
                supports_diesel=True,
            )
        )
    else:
        ev_score -= 20
        diesel_score += 20
        reasons.append(
            FitReason(
                code="diesel_high_daily_mileage",
                label="High daily mileage favours diesel",
                detail=f"{daily} miles/day exceeds comfortable EV range without en-route charging.",
                supports_ev=False,
                supports_diesel=True,
            )
        )

    # --- Urban share ---
    if urban_pct >= 70:
        ev_score += 20
        diesel_score -= 10
        reasons.append(
            FitReason(
                code="ev_urban_heavy",
                label="Urban-heavy route favours EV",
                detail=f"{urban_pct}% urban mileage — stop-start suits EV regeneration and zero tailpipe emissions.",
                supports_ev=True,
                supports_diesel=False,
            )
        )
    elif urban_pct >= 40:
        ev_score += 5
        reasons.append(
            FitReason(
                code="ev_mixed_urban",
                label="Mixed urban/rural",
                detail=f"{urban_pct}% urban — EV still viable with charging access.",
                supports_ev=True,
                supports_diesel=False,
            )
        )
    else:
        diesel_score += 10
        reasons.append(
            FitReason(
                code="diesel_rural_heavy",
                label="Rural-heavy route favours diesel",
                detail=f"Only {urban_pct}% urban — longer distances between charges make diesel simpler.",
                supports_ev=False,
                supports_diesel=True,
            )
        )

    # --- Charging access ---
    if charging_access:
        ev_score += 20
        reasons.append(
            FitReason(
                code="ev_charging_access",
                label="Charging access available",
                detail="Home or depot charging means the EV can start each day full.",
                supports_ev=True,
                supports_diesel=False,
            )
        )
    else:
        ev_score -= 25
        diesel_score += 10
        reasons.append(
            FitReason(
                code="ev_no_charging_access",
                label="No home/depot charging",
                detail="Without dedicated charging, keeping an EV van fuelled is difficult for a tradesperson.",
                supports_ev=False,
                supports_diesel=True,
            )
        )

    # --- Payload / van size ---
    if payload >= 1400 or applicant.van_size == "large":
        ev_score -= 10
        diesel_score += 10
        reasons.append(
            FitReason(
                code="diesel_heavy_payload",
                label="Heavy payload favours diesel",
                detail=f"{payload} kg payload / {applicant.van_size} van — electric payload penalties reduce usable load.",
                supports_ev=False,
                supports_diesel=True,
            )
        )
    elif payload <= 800:
        ev_score += 5
        reasons.append(
            FitReason(
                code="ev_light_payload",
                label="Light payload suits EV",
                detail=f"{payload} kg payload is well within EV van capacity.",
                supports_ev=True,
                supports_diesel=False,
            )
        )

    # --- Primary use ---
    if primary_use == "delivery":
        ev_score += 10
        diesel_score -= 5
        reasons.append(
            FitReason(
                code="ev_delivery_stop_start",
                label="Multi-drop delivery suits EV",
                detail="Frequent stops favour EV regeneration and low-speed efficiency.",
                supports_ev=True,
                supports_diesel=False,
            )
        )
    elif primary_use == "tradesman_transport":
        diesel_score += 5
        reasons.append(
            FitReason(
                code="diesel_tradesman_tool_load",
                label="Tradesman tool carrying",
                detail="Heavy tools and site work can strain EV range/payload margins.",
                supports_ev=False,
                supports_diesel=True,
            )
        )

    # --- Budget ---
    # Both estimates come from the SHARED cost model, so the figure quoted
    # here is exactly the figure the credit engine scores affordability
    # against.
    ev_cost = van_monthly_cost(
        "ev", daily, charging_access=charging_access, urban_mileage_pct=urban_pct
    )
    diesel_cost = van_monthly_cost(
        "diesel", daily, charging_access=charging_access, urban_mileage_pct=urban_pct
    )
    est_ev_monthly = ev_cost.total
    est_diesel_monthly = diesel_cost.total

    if budget > 0:
        if est_ev_monthly <= budget:
            ev_score += 5
            reasons.append(
                FitReason(
                    code="ev_within_budget",
                    label="EV within monthly budget",
                    detail=f"Estimated £{est_ev_monthly:.0f}/month vs £{budget} budget.",
                    supports_ev=True,
                    supports_diesel=False,
                )
            )
        else:
            ev_score -= 10
            reasons.append(
                FitReason(
                    code="ev_over_budget",
                    label="EV above monthly budget",
                    detail=f"Estimated £{est_ev_monthly:.0f}/month vs £{budget} budget.",
                    supports_ev=False,
                    supports_diesel=True,
                )
            )
        if est_diesel_monthly <= budget:
            diesel_score += 5
        else:
            diesel_score -= 5

    ev_score = max(0.0, min(100.0, ev_score))
    diesel_score = max(0.0, min(100.0, diesel_score))

    # --- Recommendation ---
    margin = abs(ev_score - diesel_score)
    if ev_score > diesel_score + 8:
        recommendation: VanFit = "ev"
        confidence = min(0.95, 0.55 + margin / 220)
        note = (
            f"Recommend EV van: {daily} miles/day, {urban_pct}% urban, "
            f"{'charging available' if charging_access else 'no charging'}. "
            f"Estimated £{est_ev_monthly:.0f}/month."
        )
    elif diesel_score > ev_score + 8:
        recommendation = "diesel"
        confidence = min(0.95, 0.55 + margin / 220)
        note = (
            f"Recommend diesel van: {daily} miles/day, "
            f"{'no dedicated charging' if not charging_access else 'charging available'}. "
            f"Estimated £{est_diesel_monthly:.0f}/month."
        )
    else:
        recommendation = "hybrid"
        confidence = 0.55
        note = (
            f"Either EV or diesel could work ({ev_score:.0f} vs {diesel_score:.0f}). "
            f"Consider charging access and payload before deciding."
        )

    return EVVsDieselFit(
        recommendation=recommendation,
        confidence=round(confidence, 2),
        ev_score=round(ev_score, 1),
        diesel_score=round(diesel_score, 1),
        reasons=reasons,
        est_monthly_ev_gbp=round(est_ev_monthly, 0),
        est_monthly_diesel_gbp=round(est_diesel_monthly, 0),
        note=note,
    )
