"""Unified cost model and integrated affordability assessment.

WHY THIS MODULE EXISTS
----------------------
The credit engine and the vehicle-fit engine used to disagree about what a
van costs:

* ``ml_engine.applicant_to_feature_row`` assumed ``180 + 180 * vans`` per
  month (so ~GBP 360/month for one van) and used that for both the DTI
  feature and the ``installment`` feature.
* ``fit_finder`` computed ``lease + fuel + maintenance + road tax + tolls``,
  which came out at roughly GBP 500-1,274/month.

So the credit model was scoring affordability against a number that ignored
fuel, maintenance and tax, while the fit finder was quoting a number the
credit model had never seen. A business could be "approved" on affordability
and simultaneously have no vehicle it could actually afford.

THE FIX
-------
There is now ONE set of cost assumptions, defined here, and both engines
import from it:

* ``fit_finder`` quotes the same per-van monthly cost that
  ``ml_engine`` puts into the credit model's ``installment``/``dti``
  features, and
* the integrated affordability verdict (below) compares that same cost,
  multiplied by the number of vans requested, against the applicant's
  turnover and stated budget.

Only the *scoring* feature is clipped to the training data's range (the
model was trained on consumer loans, so feeding it a GBP 1,200 instalment
would be extrapolation). The *affordability* figures shown to the user are
always the true, unclipped costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal

import numpy as np

# ---------------------------------------------------------------------------
# Unified per-van cost assumptions (GBP per month, per van)
# ---------------------------------------------------------------------------

# Lease instalments. These are commercial van lease payments, already
# inclusive of finance charges, so they are the amount the business actually
# commits to each month.
EV_LEASE_PER_MONTH = 420.0
DIESEL_LEASE_PER_MONTH = 320.0
HYBRID_LEASE_PER_MONTH = 370.0

# Energy / fuel per mile.
EV_ENERGY_PER_MILE = 0.14        # home or depot charging
DIESEL_FUEL_PER_MILE = 0.22
HYBRID_ENERGY_PER_MILE = 0.11    # blended charge + fuel

# Scheduled maintenance per month.
EV_MAINTENANCE_PER_MONTH = 35.0
DIESEL_MAINTENANCE_PER_MONTH = 65.0
HYBRID_MAINTENANCE_PER_MONTH = 50.0

# Annual road tax charged monthly.
EV_ROAD_TAX = 0.0
DIESEL_ROAD_TAX = 15.0
HYBRID_ROAD_TAX = 15.0

# Central-London charges.
CONGESTION_CHARGE = 15.0        # avoided by EV in central London zones
ULEZ = 12.50                    # avoided by EV

DAYS_PER_MONTH = 22

# ---------------------------------------------------------------------------
# Credit-affordability policy
# ---------------------------------------------------------------------------

# The share of monthly turnover that may be committed to van running costs.
# 15% is a deliberately conservative SME van-leasing guide: it leaves enough
# headroom for materials, wages and tax even in a good month.
MAX_COMMITMENT_PCT_OF_TURNOVER = 15.0

# Minimum headroom required between the applicant's stated budget and the
# cost of the recommended vehicle before the deal is called "workable".
MIN_BUDGET_HEADROOM_GBP = 50.0


Powertrain = Literal["ev", "diesel", "hybrid"]

# The order below is also the preference order used when nothing else
# distinguishes the options.
POWERTRAINS: List[Powertrain] = ["ev", "diesel", "hybrid"]


@dataclass
class VanCost:
    """A per-van monthly running cost, itemised."""

    powertrain: Powertrain
    lease: float
    energy: float
    maintenance: float
    road_tax: float
    tolls_and_charges: float

    @property
    def total(self) -> float:
        return (
            self.lease
            + self.energy
            + self.maintenance
            + self.road_tax
            + self.tolls_and_charges
        )

    def breakdown(self) -> Dict[str, float]:
        return {
            "Lease": self.lease,
            "Energy / fuel": self.energy,
            "Maintenance": self.maintenance,
            "Road tax": self.road_tax,
            "Charges & tolls": self.tolls_and_charges,
        }



def van_monthly_cost(
    powertrain: Powertrain,
    daily_mileage: int,
    charging_access: bool = False,
    urban_mileage_pct: int = 0,
) -> VanCost:
    """Return the itemised monthly running cost of ONE van.

    The same function backs the credit engine's affordability features and
    the fit-finder's quoted costs, so the two can never disagree.
    """
    daily = max(int(daily_mileage), 0)
    miles_per_month = daily * DAYS_PER_MONTH

    if powertrain == "ev":
        lease = EV_LEASE_PER_MONTH
        per_mile = EV_ENERGY_PER_MILE
        maintenance = EV_MAINTENANCE_PER_MONTH
        road_tax = EV_ROAD_TAX
        charges = 0.0 if charging_access else CONGESTION_CHARGE + ULEZ
    elif powertrain == "diesel":
        lease = DIESEL_LEASE_PER_MONTH
        per_mile = DIESEL_FUEL_PER_MILE
        maintenance = DIESEL_MAINTENANCE_PER_MONTH
        road_tax = DIESEL_ROAD_TAX
        charges = (CONGESTION_CHARGE + ULEZ) if urban_mileage_pct >= 50 else 0.0
    else:  # hybrid
        lease = HYBRID_LEASE_PER_MONTH
        per_mile = HYBRID_ENERGY_PER_MILE
        maintenance = HYBRID_MAINTENANCE_PER_MONTH
        road_tax = HYBRID_ROAD_TAX
        charges = (CONGESTION_CHARGE + ULEZ) if urban_mileage_pct >= 50 else 0.0

    return VanCost(
        powertrain=powertrain,
        lease=lease,
        energy=round(per_mile * miles_per_month, 2),
        maintenance=maintenance,
        road_tax=road_tax,
        tolls_and_charges=charges,
    )


def credit_aware_van_cost(
    powertrain: Powertrain,
    daily_mileage: int,
    charging_access: bool = False,
    urban_mileage_pct: int = 0,
) -> float:
    """Per-van monthly cost used INSIDE the credit model.

    The credit model was trained on consumer loans whose instalment feature
    has a p99 of ~GBP 870/month. Feeding it a much larger number would be
    extrapolation outside the training distribution, so the value used for
    scoring is clipped to that range.

    IMPORTANT: the clip applies to the SCORING feature only. The figures
    shown to the user always come from :func:`van_monthly_cost`, which is
    unclipped.
    """
    return float(
        np.clip(van_monthly_cost(powertrain, daily_mileage, charging_access,
                                 urban_mileage_pct).total, 30.0, 900.0)
    )



# ---------------------------------------------------------------------------
# Integrated affordability verdict
# ---------------------------------------------------------------------------

AffordabilityVerdict = Literal["workable", "tight", "not_workable"]


@dataclass
class AffordabilityResult:
    """The answer to "can this deal actually work?".

    Combines the credit decision with the budget fit and the vehicle
    recommendation into a single verdict, and produces concrete next
    actions rather than leaving the user to reconcile three screens.
    """

    verdict: AffordabilityVerdict
    headline: str
    monthly_cost_per_van: float
    monthly_cost_total: float
    monthly_turnover: float
    commitment_pct: float          # cost as % of monthly turnover
    budget: float                  # applicant's stated budget (0 = unknown)
    budget_gap: float              # budget - monthly_cost_total (negative = over)
    within_budget: bool
    recommended_powertrain: str
    reasons: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)

    @property
    def is_workable(self) -> bool:
        """True only for a clean 'workable' verdict (not 'tight')."""
        return self.verdict == "workable"


# ---------------------------------------------------------------------------
# Applicant-level helpers
# ---------------------------------------------------------------------------
# These bridge the shared cost model to the Applicant dataclass so that the
# credit engine, the fit finder and the affordability verdict all read the
# same usage assumptions off the same object.

def applicant_charging_access(applicant) -> bool:
    """True when the applicant can charge on site (home or depot)."""
    return bool(
        getattr(applicant, "home_charging", False)
        or getattr(applicant, "depot_charging", False)
    )


def applicant_van_cost(applicant, powertrain: Powertrain) -> VanCost:
    """Itemised UNCLIPPED monthly cost of one van for this applicant."""
    return van_monthly_cost(
        powertrain,
        int(getattr(applicant, "daily_mileage", 0) or 0),
        charging_access=applicant_charging_access(applicant),
        urban_mileage_pct=int(getattr(applicant, "urban_mileage_pct", 0) or 0),
    )


def applicant_credit_aware_cost(applicant, powertrain: Powertrain) -> float:
    """Per-van monthly cost used INSIDE the credit model (clipped)."""
    return credit_aware_van_cost(
        powertrain,
        int(getattr(applicant, "daily_mileage", 0) or 0),
        charging_access=applicant_charging_access(applicant),
        urban_mileage_pct=int(getattr(applicant, "urban_mileage_pct", 0) or 0),
    )


def applicant_total_van_cost(applicant, powertrain: Powertrain) -> float:
    """UNCLIPPED total monthly van cost for all requested vans."""
    vans = max(int(getattr(applicant, "requested_vans", 1) or 1), 1)
    return applicant_van_cost(applicant, powertrain).total * vans


def powertrain_for_affordability(applicant, recommendation: str) -> str:
    """Choose the powertrain whose cost the affordability verdict should use.

    The fit finder returns "hybrid" when neither EV nor diesel is clearly
    better. That is a genuine "pick one" recommendation, so the affordability
    verdict should be based on the CHEAPER of the two - that is the best
    case available to the applicant. Using the dearer option would fail
    deals that are actually affordable, and using the credit engine's
    internal hybrid reference would quote a vehicle the fit finder never
    recommended.
    """
    if recommendation in ("ev", "diesel"):
        return recommendation

    ev = applicant_van_cost(applicant, "ev").total
    diesel = applicant_van_cost(applicant, "diesel").total
    return "ev" if ev <= diesel else "diesel"


def assess_applicant_deal(applicant, decision: str, recommendation: str):
    """Run the full affordability assessment for one applicant.

    This is the single entry point the Streamlit app uses, so the verdict
    shown to the user is always computed from the same cost model that the
    credit engine scored against.
    """
    powertrain = powertrain_for_affordability(applicant, recommendation)
    per_van = applicant_van_cost(applicant, powertrain).total
    return assess_affordability(applicant, decision, powertrain, per_van)



def assess_affordability(
    applicant,
    decision: str,
    recommended_powertrain: str,
    per_van_cost: float,
) -> AffordabilityResult:
    """Combine credit decision, budget fit and vehicle fit into one verdict.

    ``per_van_cost`` should be the UNCLIPPED monthly running cost of one van
    of the recommended type (see :func:`van_monthly_cost`).

    The verdict combines three independent questions:

    1. **Can they pay?** - is the total van cost affordable against monthly
       turnover, and against the stated budget if one was given?
    2. **Will they be approved?** - the credit decision from the ML engine.
    3. **Does it fit?** - the vehicle the fit-finder recommended.

    A deal is only "workable" when all three line up.
    """
    vans = max(int(getattr(applicant, "requested_vans", 1) or 1), 1)
    turnover = float(getattr(applicant, "annual_turnover_gbp", 0.0) or 0.0)
    monthly_turnover = turnover / 12.0
    budget = float(getattr(applicant, "monthly_budget_gbp", 0.0) or 0.0)

    total = per_van_cost * vans
    commitment_pct = (total / monthly_turnover * 100.0) if monthly_turnover > 0 else 0.0

    within_budget = budget > 0 and total <= budget
    budget_gap = (budget - total) if budget > 0 else 0.0

    # --- Component checks -------------------------------------------------
    turnover_ok = commitment_pct <= MAX_COMMITMENT_PCT_OF_TURNOVER
    budget_ok = (budget <= 0) or within_budget

    reasons: List[str] = []
    actions: List[str] = []

    if not turnover_ok:
        reasons.append(
            f"Van costs are {commitment_pct:.0f}% of monthly turnover, above the "
            f"{MAX_COMMITMENT_PCT_OF_TURNOVER:.0f}% guideline for SME van leasing."
        )
    else:
        reasons.append(
            f"Van costs are {commitment_pct:.0f}% of monthly turnover, within the "
            f"{MAX_COMMITMENT_PCT_OF_TURNOVER:.0f}% guideline."
        )

    if budget > 0:
        if within_budget:
            reasons.append(
                f"Total of {vans} van(s) at GBP {total:,.0f}/month sits GBP "
                f"{budget_gap:,.0f}/month under the stated budget of GBP {budget:,.0f}."
            )
        else:
            reasons.append(
                f"Total of {vans} van(s) at GBP {total:,.0f}/month exceeds the "
                f"stated budget of GBP {budget:,.0f} by GBP {abs(budget_gap):,.0f}/month."
            )
    else:
        reasons.append("No monthly budget was stated, so budget fit was not assessed.")

    if decision == "approve":
        reasons.append("The credit model approves this application.")
    elif decision == "refer":
        reasons.append("The credit model refers this application for manual review.")
    else:
        reasons.append("The credit model declines this application.")

    # --- Overall verdict --------------------------------------------------
    if decision == "decline" or not turnover_ok or not budget_ok:
        verdict: AffordabilityVerdict = "not_workable"
        headline = "This deal does not work as described."
    elif decision == "refer" or (budget > 0 and budget_gap < MIN_BUDGET_HEADROOM_GBP):
        verdict = "tight"
        headline = "This deal is workable but tight - it needs a closer look."
    else:
        verdict = "workable"
        headline = "This deal works."

    # --- Concrete next actions -------------------------------------------
    if verdict == "not_workable":
        if decision == "decline":
            actions.append(
                "Address the credit issues flagged above, or reduce the number of vans "
                "requested, then re-run the assessment."
            )
        if not turnover_ok:
            reduction = (commitment_pct - MAX_COMMITMENT_PCT_OF_TURNOVER) / 100.0 * monthly_turnover
            affordable_vans = max(
                1, int(monthly_turnover * MAX_COMMITMENT_PCT_OF_TURNOVER / 100.0 / per_van_cost)
            )
            actions.append(
                f"Reduce the commitment by about GBP {reduction:,.0f}/month - for this "
                f"turnover, {affordable_vans} van(s) at GBP {per_van_cost:,.0f} is "
                f"about the guideline limit."
            )
        if not budget_ok:
            actions.append(
                f"Either raise the budget to at least GBP {total:,.0f}/month, or move to "
                f"a cheaper vehicle option."
            )
    elif verdict == "tight":
        actions.append(
            "Route to manual review - confirm the affordability holds in a quiet month, "
            "not just a busy one."
        )
        if budget > 0 and budget_gap < MIN_BUDGET_HEADROOM_GBP:
            actions.append(
                f"Headroom is only GBP {budget_gap:,.0f}/month. Consider a cheaper powertrain "
                f"or fewer vans to create a buffer."
            )
    else:
        actions.append(
            f"Proceed with {vans} van(s) at the recommended {recommended_powertrain} "
            f"option - GBP {total:,.0f}/month."
        )
        actions.append(
            "Confirm the finance agreement and insurance are in place before delivery."
        )

    if decision == "refer":
        actions.insert(0, "A human underwriter needs to review this before approval.")

    return AffordabilityResult(
        verdict=verdict,
        headline=headline,
        monthly_cost_per_van=per_van_cost,
        monthly_cost_total=total,
        monthly_turnover=monthly_turnover,
        commitment_pct=commitment_pct,
        budget=budget,
        budget_gap=budget_gap,
        within_budget=within_budget,
        recommended_powertrain=recommended_powertrain,
        reasons=reasons,
        actions=actions,
    )
