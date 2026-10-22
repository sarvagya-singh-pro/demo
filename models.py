"""Shared data models for the underwriting consistency tester and van fit finder.

All models are plain dataclasses so the demo has no heavy ORM or framework
dependency beyond pandas/streamlit. Decisions are framed as *vetting* outcomes
(approve / refer / decline), never credit scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

Decision = Literal["approve", "refer", "decline"]
VanFit = Literal["ev", "diesel", "hybrid"]


@dataclass
class Applicant:
    """A synthetic SME van-subscription applicant."""

    business_name: str
    trade: str                      # free-text trade description
    trade_category: str             # engine-normalised risk tier
    years_in_business: float
    employee_count: int
    annual_turnover_gbp: float
    adverse_credit: bool            # CCJ / default / etc.
    fleet_size: int                 # vans already operated
    requested_vans: int
    primary_use: str                # "tradesman_transport" | "delivery" | "mixed"
    urban_mileage_pct: int          # 0-100
    daily_mileage: int              # miles per day
    home_charging: bool = False
    depot_charging: bool = False
    payload_kg: int = 0
    van_size: str = "medium"        # small | medium | large
    monthly_budget_gbp: int = 0

    def to_dict(self) -> dict:
        return {
            "business_name": self.business_name,
            "trade": self.trade,
            "trade_category": self.trade_category,
            "years_in_business": self.years_in_business,
            "employee_count": self.employee_count,
            "annual_turnover_gbp": self.annual_turnover_gbp,
            "adverse_credit": self.adverse_credit,
            "fleet_size": self.fleet_size,
            "requested_vans": self.requested_vans,
            "primary_use": self.primary_use,
            "urban_mileage_pct": self.urban_mileage_pct,
            "daily_mileage": self.daily_mileage,
            "home_charging": self.home_charging,
            "depot_charging": self.depot_charging,
            "payload_kg": self.payload_kg,
            "van_size": self.van_size,
            "monthly_budget_gbp": self.monthly_budget_gbp,
        }


@dataclass
class DecisionResult:
    """Output of the underwriting engine for a single applicant description."""

    decision: Decision
    confidence: float               # 0-1, how far from the decision boundary
    reasons: List[str]              # plain-English reason codes
    reason_details: List[str]       # human-readable explanation per code
    score: float                    # internal vetting score (not a credit score)
    threshold: float
    applicant: Applicant


@dataclass
class ParaphraseResult:
    """One paraphrase of the same underlying business facts."""

    label: str                      # e.g. "original", "paraphrase_1"
    trade_description: str          # how the trade was worded
    decision: DecisionResult
    flipped: bool                   # decision != base decision


@dataclass
class ConsistencyReport:
    """Aggregate result of running an applicant through N paraphrases."""

    applicant: Applicant
    base_decision: Decision
    base_reasons: List[str]
    total_paraphrases: int
    flip_count: int
    flip_rate: float                # 0-1
    decision_distribution: dict     # decision -> count
    reason_stability_rate: float    # 0-1, share of paraphrases with identical reason set
    paraphrases: List[ParaphraseResult]


@dataclass
class FitReason:
    code: str
    label: str
    detail: str
    supports_ev: bool
    supports_diesel: bool


@dataclass
class EVVsDieselFit:
    """EV vs diesel van fit recommendation."""

    recommendation: VanFit
    confidence: float               # 0-1
    ev_score: float                 # 0-100 suitability for electric
    diesel_score: float             # 0-100 suitability for diesel
    reasons: List[FitReason]
    est_monthly_ev_gbp: float
    est_monthly_diesel_gbp: float
    note: str                       # plain-English summary