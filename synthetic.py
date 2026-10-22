"""Synthetic SME applicant generator + paraphrase (re-wording) engine.

Produces realistic London tradespeople/SME applicants (1-10 staff, small fleets)
and multiple wordings of the *same* underlying facts so the consistency tester
can measure how often a decision flips purely because of wording.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from models import Applicant
from affordability import van_monthly_cost

# ---------------------------------------------------------------------------
# Trade vocabularies: same underlying trade, different wordings
# ---------------------------------------------------------------------------

TRADE_VARIANTS = {
    "electrician": [
        "electrician",
        "electrical contractor",
        "domestic electrician",
        "electrical installation",
    ],
    "plumber": [
        "plumber",
        "plumbing & heating engineer",
        "domestic plumber",
        "plumbing services",
    ],
    "builder": [
        "builder",
        "general building contractor",
        "residential builder",
        "building works",
    ],
    "handyman": [
        "handyman",
        "odd-job person",
        "property maintenance",
        "home repairs",
    ],
    "gardener": [
        "gardener",
        "landscape gardener",
        "garden maintenance",
        "groundskeeping",
    ],
    "cleaner": [
        "cleaner",
        "commercial cleaning",
        "office cleaning services",
        "cleaning contractor",
    ],
    "demolition": [
        "demolition contractor",
        "site clearance",
        "demolition & wrecking",
        "structural demolition",
    ],
    "waste": [
        "waste collection",
        "waste management",
        "rubbish removal",
        "waste disposal services",
    ],
}

# Business-name stems per trade
NAME_STEMS = {
    "electrician": ["Spark", "Volt", "Current", "Wire"],
    "plumber": ["Flow", "Pipe", "Tap", "Heat"],
    "builder": ["Brick", "Build", "Frame", "Construct"],
    "handyman": ["FixIt", "Handy", "Mend", "Care"],
    "gardener": ["Green", "Leaf", "Grow", "Lawn"],
    "cleaner": ["Shine", "Clean", "Sweep", "Polish"],
    "demolition": ["Wreck", "Raze", "Clear", "Demo"],
    "waste": ["Bin", "Tip", "Haul", "Clear"],
}

PRIMARY_USES = ["tradesman_transport", "delivery", "mixed"]
VAN_SIZES = ["small", "medium", "large"]

LOCATIONS = [
    "London", "Croydon", "Bromley", "Enfield", "Hounslow", "Ilford",
    "Romford", "Watford", "Slough", "Dartford",
]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def generate_applicant(seed: int = 42) -> Applicant:
    """Generate one synthetic SME applicant with a reproducible seed.

    ECONOMIC COHERENCE
    ------------------
    The applicant must be able to plausibly carry the vans it is asking for.
    A van genuinely costs GBP 560-1,150/month depending on powertrain and
    mileage (see ``affordability.py``), so a business leasing 3 vans needs
    roughly GBP 2,000-3,500/month of headroom before it is "tight".

    The previous generator drew turnover, van count and budget independently,
    which produced applicants that could never work: the median applicant
    had GBP 3,708/month of turnover and 3 vans costing GBP 2,500-3,500/month
    - a 70-95% commitment against a 15% guideline. Every single applicant
    came back "not workable", so the affordability verdict carried no
    information.

    Turnover is therefore DERIVED from the van cost and a target commitment
    ratio drawn from a spread, so the population deliberately contains
    workable, tight and unaffordable cases.
    """
    r = _rng(seed)
    trade_key = r.choice(list(TRADE_VARIANTS.keys()))
    trade = r.choice(TRADE_VARIANTS[trade_key])
    name_stem = r.choice(NAME_STEMS[trade_key])
    location = r.choice(LOCATIONS)

    years = round(r.uniform(0.3, 6.0), 1)
    employees = r.randint(1, 10)
    adverse = r.random() < 0.35
    fleet_size = r.randint(0, 6)
    requested = r.randint(1, 4)
    # Staffing: sometimes deliberately mismatched to create decline/refer cases
    if r.random() < 0.45:
        employees = r.randint(1, 2)  # under-staffed for fleet requested
    primary_use = r.choice(PRIMARY_USES)
    urban_pct = r.randint(30, 100)
    daily_mileage = r.randint(20, 220)
    home_charging = r.random() < 0.55
    depot_charging = r.random() < 0.35
    payload = r.choice([500, 800, 1200, 1600])
    van_size = r.choice(VAN_SIZES)

    # --- Economics, derived from the real cost model -------------------
    # Size the business against the cost of the vans it is asking for.
    #
    # The business is sized against the DEARER of the two realistic
    # powertrains (EV vs diesel), so it can carry whichever option the fit
    # finder ends up recommending. Sizing against a mid-point would leave
    # the deal unaffordable whenever the dearer option was picked.
    charging_access = home_charging or depot_charging

    ev_cost = van_monthly_cost(
        "ev", daily_mileage, charging_access, urban_pct
    ).total
    diesel_cost = van_monthly_cost(
        "diesel", daily_mileage, charging_access, urban_pct
    ).total
    reference_per_van = max(ev_cost, diesel_cost)
    total_van_cost = reference_per_van * requested

    # Target share of monthly turnover committed to vans. The policy
    # guideline is 15%. The mix below is weighted towards comfortable
    # cases, because in practice applicants reaching underwriting have
    # usually been pre-screened for affordability - but a meaningful
    # minority are tight or genuinely unaffordable.
    #
    #   ~55% comfortable (below the guideline)
    #   ~25% tight (around the guideline)
    #   ~20% unaffordable (well above it)
    ratio_roll = r.random()
    if ratio_roll < 0.55:
        commitment_ratio = r.uniform(0.05, 0.12)
    elif ratio_roll < 0.80:
        commitment_ratio = r.uniform(0.12, 0.17)
    else:
        commitment_ratio = r.uniform(0.17, 0.28)

    monthly_turnover = total_van_cost / commitment_ratio
    turnover = round(monthly_turnover * 12, -2)

    # The stated budget is the TOTAL monthly budget for the whole deal.
    # Most applicants budget generously enough; some are optimistic and
    # will find the deal over budget.
    budget_roll = r.random()
    if budget_roll < 0.55:
        budget_multiplier = r.uniform(1.00, 1.20)
    elif budget_roll < 0.80:
        budget_multiplier = r.uniform(0.88, 1.00)
    else:
        budget_multiplier = r.uniform(1.20, 1.45)

    budget = int(round(total_van_cost * budget_multiplier, -1))

    business_name = f"{name_stem} {location} Ltd"

    return Applicant(
        business_name=business_name,
        trade=trade,
        trade_category="",  # engine will normalise
        years_in_business=years,
        employee_count=employees,
        annual_turnover_gbp=turnover,
        adverse_credit=adverse,
        fleet_size=fleet_size,
        requested_vans=requested,
        primary_use=primary_use,
        urban_mileage_pct=urban_pct,
        daily_mileage=daily_mileage,
        home_charging=home_charging,
        depot_charging=depot_charging,
        payload_kg=payload,
        van_size=van_size,
        monthly_budget_gbp=budget,
    )


# ---------------------------------------------------------------------------
# Paraphrase generation
# ---------------------------------------------------------------------------

TRADE_REWORD_TEMPLATES = [
    "We are a {trade} based in {location}.",
    "{trade} operating out of {location}.",
    "Sole trader {trade} serving {location} and surrounding areas.",
    "{trade} business in {location}.",
    "Providing {trade} services across {location}.",
]


def paraphrase_trade(trade: str, location: str, variant: str) -> str:
    """Re-word the same trade into a different phrasing (same facts)."""
    template = TRADE_REWORD_TEMPLATES[
        abs(hash((trade, location, variant))) % len(TRADE_REWORD_TEMPLATES)
    ]
    return template.format(trade=variant, location=location)


def generate_paraphrases(
    base: Applicant, n: int = 5, seed: int = 7
) -> List[Tuple[str, str]]:
    """Return [(label, re-worded trade description)] for the same business.

    The underlying facts (years, turnover, staff, credit, fleet) are identical;
    only the *wording* of the trade changes. This is what can flip a keyword-
    sensitive underwriting engine.
    """
    r = _rng(seed)
    trade_key = None
    for key, variants in TRADE_VARIANTS.items():
        if base.trade in variants:
            trade_key = key
            break
    if trade_key is None:
        trade_key = base.trade.lower().split()[0]

    variants = TRADE_VARIANTS.get(trade_key, [base.trade])
    parts = base.business_name.split()
    location = parts[-2] if len(parts) >= 3 else ""

    paraphrases: List[Tuple[str, str]] = [("original", base.trade)]
    chosen = (
        r.sample(variants, min(n - 1, len(variants)))
        if len(variants) >= n - 1
        else variants
    )
    # pad by re-sampling with replacement if we need more than we have variants
    while len(chosen) < n - 1:
        chosen.append(r.choice(variants))

    for i, variant in enumerate(chosen, start=1):
        paraphrases.append((f"paraphrase_{i}", paraphrase_trade(variant, location, variant)))
    return paraphrases


def apply_paraphrase(base: Applicant, trade_description: str) -> Applicant:
    """Copy the applicant with a re-worded trade (facts unchanged)."""
    import copy

    app = copy.deepcopy(base)
    app.trade = trade_description
    app.trade_category = ""
    return app

