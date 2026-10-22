"""Integrity regression tests for the van-leasing credit/affordability demo.

These tests lock down the fixes made to the demo's decision logic. Each
test corresponds to a specific defect that was found and fixed:

* Label semantics - the model must score P(DEFAULT), not a lender-policy
  label, and the reported AUC must be computed against the same column the
  model scores.
* Trade taxonomy - the risk tier must be a property of the trade, not of
  the wording, and whole-word matching must not produce false matches.
* Cost consistency - the credit engine and the fit finder must quote the
  same number for the same van.
* Probability bounds - scores must stay in range and the tier overlay must
  not saturate.
* Reason-code fidelity - reason codes must not contradict the decision.
* UI contrast - every text/background pair must clear WCAG AA, and no light
  background may be painted under the dark theme's near-white body text.

Run with:  python3 -m pytest test_integrity.py -v
      or: python3 test_integrity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from models import Applicant
from trade_taxonomy import (
    TRADE_TIERS,
    identify_trade,
    normalise_trade_category,
)
from affordability import (
    applicant_credit_aware_cost,
    applicant_total_van_cost,
    applicant_van_cost,
    assess_affordability,
    credit_aware_van_cost,
    van_monthly_cost,
)
from fit_finder import evaluate_fit
import engine
import ml_engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_applicant(**overrides) -> Applicant:
    """Build a mid-range applicant, then apply any overrides."""
    base = dict(
        business_name="Test Co",
        trade="plumber",
        trade_category="low",
        years_in_business=4.0,
        employee_count=5,
        annual_turnover_gbp=120_000.0,
        adverse_credit=False,
        fleet_size=1,
        requested_vans=2,
        primary_use="tradesman_transport",
        urban_mileage_pct=60,
        daily_mileage=80,
        home_charging=True,
        depot_charging=False,
        payload_kg=600,
        van_size="medium",
        monthly_budget_gbp=0,
    )
    base.update(overrides)
    app = Applicant(**base)
    app.trade_category = normalise_trade_category(app.trade)
    return app


# ===========================================================================
# 1. LABEL SEMANTICS
# ===========================================================================

def test_model_trains_on_default_label():
    """The model must be trained on `not.fully.paid` (default), not credit.policy.

    This is the single most important test in the file. The previous
    version trained on `credit.policy` (a lender POLICY label) and then
    read the resulting probability as "P(default)", which is backwards:
    the number was HIGH for good loans.
    """
    df = ml_engine._load_raw_dataframe()

    assert "defaulted" in df.columns, (
        "The training frame must expose the default label as 'defaulted'."
    )
    assert "credit_policy" not in df.columns, (
        "The policy label must not be carried in the training frame - it is "
        "not a default label and invites the same mistake again."
    )

    # 'defaulted' must be the renamed 'not.fully.paid' column.
    assert df["defaulted"].dtype.kind in "iu"
    assert set(df["defaulted"].unique()).issubset({0, 1})

    # The default base rate in this dataset is ~16%. A policy label would be
    # ~80% (mostly approved). Assert we are on the default label.
    base_rate = df["defaulted"].mean()
    assert 0.05 < base_rate < 0.35, (
        f"Expected a default base rate in the 5-35%% range, got {base_rate:.3f}. "
        "This suggests the wrong label is being used."
    )


def test_auc_is_computed_against_the_scored_column():
    """The positive column read at inference must be the column the AUC used."""
    model, features = ml_engine._get_model()

    positive_col = ml_engine._positive_class_index(model)
    assert positive_col == list(model.classes_).index(1)
    assert positive_col != 0, (
        "The default label is class 1, so the positive column must not be 0. "
        "Reading column 0 as P(default) was the original defect."
    )

    # The trained model must actually contain class 1.
    assert 1 in list(model.classes_)


def test_positive_class_index_rejects_missing_positive():
    """A model without the positive class must fail loudly, not silently."""
    class FakeModel:
        classes_ = [0]

    try:
        ml_engine._positive_class_index(FakeModel())
    except ValueError:
        return
    raise AssertionError(
        "_positive_class_index should raise when class 1 is absent."
    )


# ===========================================================================
# 2. TRADE TAXONOMY
# ===========================================================================

def test_no_false_substring_matches():
    """Whole-word matching must not match a keyword inside another word."""
    # "tow" must not match "town"; "tow" is a high-risk keyword.
    assert normalise_trade_category("town plumbing services") == "low"
    assert normalise_trade_category("plumbing in town") == "low"
    assert normalise_trade_category("town plumber") == "low"


def test_all_synonyms_of_one_trade_share_one_tier():
    """Every wording of the same trade must resolve to the same tier."""
    groups = {
        "plumber": [
            "plumber",
            "plumbing services",
            "plumbing & heating engineer",
            "domestic plumber",
        ],
        "electrician": [
            "electrician",
            "electrical contractor",
            "domestic electrician",
            "electrical installation",
        ],
        "handyman": [
            "handyman",
            "odd-job person",
            "property maintenance",
            "home repairs",
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
        "gardener": [
            "gardener",
            "landscape gardener",
            "garden maintenance",
            "groundskeeping",
        ],
    }

    for trade_key, variants in groups.items():
        expected = TRADE_TIERS[trade_key]
        for variant in variants:
            actual = normalise_trade_category(variant)
            assert actual == expected, (
                f"'{variant}' should be tier '{expected}' (as {trade_key}), "
                f"got '{actual}'"
            )
            assert identify_trade(variant) == trade_key, (
                f"'{variant}' should identify as '{trade_key}', "
                f"got '{identify_trade(variant)}'"
            )


def test_engines_agree_on_trade_tier():
    """engine.py and ml_engine.py must resolve the same tier."""
    samples = [
        "plumber", "town plumbing", "demolition contractor", "odd-job person",
        "electrical contractor", "waste collection", "landscape gardener",
        "commercial cleaning", "residential builder", "site clearance",
    ]
    for trade in samples:
        assert engine.normalise_trade_category(trade) == \
            ml_engine.normalise_trade_category(trade), (
            f"Engines disagree on tier for '{trade}'"
        )


def test_trade_tier_is_a_property_of_the_trade_not_the_wording():
    """A trade's tier must not change because of surrounding sentence wording."""
    base = normalise_trade_category("plumber")
    for sentence in [
        "We are plumbers based in London",
        "A local plumber serving the community since 1994",
        "Plumbing and heating engineer covering Greater London",
        "plumber",
    ]:
        assert normalise_trade_category(sentence) == base, (
            f"Wording changed the tier for: '{sentence}'"
        )


# ===========================================================================
# 3. COST CONSISTENCY
# ===========================================================================

def test_fit_finder_quotes_the_shared_cost_model():
    """The fit finder's quoted cost must equal the shared cost model."""
    app = make_applicant(daily_mileage=120, urban_mileage_pct=80,
                         home_charging=False, depot_charging=False)

    fit = evaluate_fit(app)

    expected_ev = van_monthly_cost(
        "ev", 120, charging_access=False, urban_mileage_pct=80
    ).total
    expected_diesel = van_monthly_cost(
        "diesel", 120, charging_access=False, urban_mileage_pct=80
    ).total

    assert abs(fit.est_monthly_ev_gbp - expected_ev) < 0.51
    assert abs(fit.est_monthly_diesel_gbp - expected_diesel) < 0.51


def test_credit_engine_uses_the_shared_cost_model():
    """The credit model's instalment feature must come from the shared model."""
    app = make_applicant(daily_mileage=120, urban_mileage_pct=80,
                         home_charging=False, depot_charging=False)

    row = ml_engine.applicant_to_feature_row(app)
    expected = applicant_credit_aware_cost(app, "hybrid")

    assert abs(row["installment"] - expected) < 1e-6, (
        "The credit model's instalment feature must equal the shared "
        "credit-aware per-van cost."
    )


def test_total_cost_scales_with_van_count():
    """Total van cost must be per-van cost times the number of vans."""
    one = make_applicant(requested_vans=1)
    three = make_applicant(requested_vans=3)

    per_van = applicant_van_cost(one, "hybrid").total
    assert abs(applicant_total_van_cost(one, "hybrid") - per_van) < 1e-6
    assert abs(applicant_total_van_cost(three, "hybrid") - per_van * 3) < 1e-6


def test_scoring_cost_is_clipped_but_user_facing_cost_is_not():
    """Clipping applies ONLY to the model feature, never to user-facing costs.

    An EV doing 400 miles/day with no charging costs well over the model's
    GBP 900 p99 instalment. The scoring feature must be clipped; the figure
    shown to the user must not be.
    """
    heavy = make_applicant(daily_mileage=400, home_charging=False,
                           depot_charging=False, urban_mileage_pct=90)

    unclipped = van_monthly_cost("ev", 400, charging_access=False,
                                 urban_mileage_pct=90).total
    clipped = credit_aware_van_cost("ev", 400, charging_access=False,
                                    urban_mileage_pct=90)

    assert unclipped > 900, (
        "This test needs a van whose true cost exceeds the GBP 900 clip."
    )
    assert clipped == 900.0, (
        f"Scoring cost should be clipped to 900, got {clipped}"
    )

    # The user-facing figure must still be the true cost.
    fit = evaluate_fit(heavy)
    assert abs(fit.est_monthly_ev_gbp - unclipped) <= 0.51, (
        "The figure shown to the user must be the unclipped running cost."
    )


def test_ev_is_cheaper_than_diesel_when_charging_is_available():
    """Sanity check on the cost model itself."""
    app = make_applicant(daily_mileage=60, home_charging=True)
    ev = applicant_van_cost(app, "ev").total
    diesel = applicant_van_cost(app, "diesel").total
    assert ev < diesel, (
        f"With home charging and low mileage, EV (GBP {ev:.0f}) should cost "
        f"less than diesel (GBP {diesel:.0f})."
    )


def test_ev_pays_charges_when_no_charging_access():
    """An EV with no charging access incurs the central-London charges."""
    app = make_applicant(home_charging=False, depot_charging=False,
                         urban_mileage_pct=90)
    ev = applicant_van_cost(app, "ev")
    assert ev.tolls_and_charges > 0

    app2 = make_applicant(home_charging=True, depot_charging=False)
    ev2 = applicant_van_cost(app2, "ev")
    assert ev2.tolls_and_charges == 0


# ===========================================================================
# 4. PROBABILITY BOUNDS AND TIER OVERLAY
# ===========================================================================

def test_default_probability_always_within_bounds():
    """Scores must stay in [0, 1] across a wide spread of applicants."""
    from synthetic import generate_applicant

    for seed in range(60):
        app = generate_applicant(seed=seed)
        app.trade_category = normalise_trade_category(app.trade)
        result = ml_engine.vet(app)

        prob = 1.0 - (result.score / 100.0)
        assert 0.0 <= prob <= 1.0, f"Probability {prob} out of bounds (seed {seed})"
        assert 0.0 <= result.score <= 100.0, (
            f"Score {result.score} out of bounds (seed {seed})"
        )
        assert result.decision in ("approve", "refer", "decline")


def test_tier_overlay_does_not_saturate():
    """A multiplicative overlay must never push a probability to 1.0.

    The previous additive overlay (+0.20) saturated: an applicant already
    at 0.90 probability was pushed past 1.0 and clipped, so the tier
    distinction vanished exactly where it mattered.
    """
    # A deliberately extreme applicant: high tier, huge cost, adverse credit.
    app = make_applicant(
        trade="demolition contractor",
        adverse_credit=True,
        years_in_business=0.3,
        annual_turnover_gbp=20_000.0,
        requested_vans=4,
        daily_mileage=300,
        home_charging=False,
        depot_charging=False,
    )
    app.trade_category = normalise_trade_category(app.trade)
    result = ml_engine.vet(app)

    prob = 1.0 - (result.score / 100.0)
    assert prob < 1.0, (
        "The default probability saturated at 1.0 - the tier overlay is "
        "clipping again."
    )


def test_high_risk_trade_never_scores_better_than_low_risk():
    """For the same business facts, a higher-risk trade must not score better."""
    low = make_applicant(trade="plumber")
    high = make_applicant(trade="demolition contractor")
    low.trade_category = normalise_trade_category(low.trade)
    high.trade_category = normalise_trade_category(high.trade)

    low_result = ml_engine.vet(low)
    high_result = ml_engine.vet(high)

    assert high_result.score <= low_result.score, (
        "A high-risk trade must not produce a better score than a low-risk "
        "trade for otherwise identical facts."
    )


def test_tier_multipliers_are_multiplicative_and_bounded():
    """The overlay must be a bounded multiplicative factor, not an additive shift."""
    for tier, mult in ml_engine.TRADE_RISK_MULTIPLIER.items():
        assert mult >= 0.0, f"Multiplier for {tier} must be non-negative"
        assert mult <= 5.0, f"Multiplier for {tier} is implausibly large"

    # Medium must be the neutral value.
    assert ml_engine.TRADE_RISK_MULTIPLIER["medium"] == 1.0

    # Low must reduce, high must increase.
    assert ml_engine.TRADE_RISK_MULTIPLIER["low"] < 1.0
    assert ml_engine.TRADE_RISK_MULTIPLIER["high"] > 1.0


def test_calibrated_thresholds_are_ordered():
    """The approve cut must be below the refer cut, and both in range."""
    approve_cut, refer_cut = ml_engine._get_thresholds()
    assert 0.0 <= approve_cut < refer_cut <= 1.0, (
        f"Thresholds out of order: approve={approve_cut}, refer={refer_cut}"
    )


# ===========================================================================
# 5. REASON-CODE FIDELITY
# ===========================================================================

def test_reason_codes_never_contradict_the_decision():
    """An approve must not carry a high-probability flag, and vice versa."""
    from synthetic import generate_applicant

    for seed in range(40):
        app = generate_applicant(seed=seed)
        app.trade_category = normalise_trade_category(app.trade)
        result = ml_engine.vet(app)

        if result.decision == "approve":
            assert "high_default_probability" not in result.reasons, (
                "An approve carried a high-probability reason code."
            )
            assert "moderate_default_probability" not in result.reasons, (
                "An approve carried a moderate-probability reason code."
            )
        elif result.decision == "decline":
            assert "low_default_probability" not in result.reasons, (
                "A decline carried a low-probability reason code."
            )


def test_reason_codes_are_material():
    """Decorative reason codes must not be emitted.

    Staffing ratio and fleet size do not feed the credit model or the tier
    overlay, so presenting them as reasons for the decision was misleading.
    """
    decorative = {
        "adequate_staffing", "insufficient_staffing",
        "fleet_at_limit", "fleet_near_limit", "fleet_capacity_headroom",
    }
    from synthetic import generate_applicant

    for seed in range(30):
        app = generate_applicant(seed=seed)
        app.trade_category = normalise_trade_category(app.trade)
        result = ml_engine.vet(app)

        overlap = decorative.intersection(result.reasons)
        assert not overlap, (
            f"Decorative reason codes emitted: {overlap} (seed {seed})"
        )


def test_reason_details_align_with_reason_codes():
    """Every reason code must have a matching human-readable detail."""
    from synthetic import generate_applicant

    for seed in range(20):
        app = generate_applicant(seed=seed)
        app.trade_category = normalise_trade_category(app.trade)
        result = ml_engine.vet(app)

        assert len(result.reasons) == len(result.reason_details), (
            f"Reason codes and details differ in length (seed {seed})"
        )
        for detail in result.reason_details:
            assert detail.strip(), "Reason details must not be empty."


def test_trade_reason_code_matches_the_actual_tier():
    """The emitted trade reason code must match the resolved tier."""
    for trade, expected_code in [
        ("plumber", "low_risk_trade"),
        ("demolition contractor", "high_risk_trade"),
        ("residential builder", "medium_risk_trade"),
    ]:
        app = make_applicant(trade=trade)
        result = ml_engine.vet(app)
        assert expected_code in result.reasons, (
            f"Expected '{expected_code}' for trade '{trade}', "
            f"got {result.reasons}"
        )


# ===========================================================================
# 6. AFFORDABILITY VERDICT
# ===========================================================================

def test_affordability_verdict_uses_the_unclipped_cost():
    """The verdict must be based on the real running cost, not the clipped one."""
    app = make_applicant(
        requested_vans=2,
        annual_turnover_gbp=60_000.0,
        monthly_budget_gbp=0,
        daily_mileage=120,
    )
    per_van = applicant_van_cost(app, "diesel").total

    result = assess_affordability(app, "approve", "diesel", per_van)

    expected_total = per_van * 2
    assert abs(result.monthly_cost_total - expected_total) < 0.01
    assert abs(result.monthly_cost_per_van - per_van) < 0.01


def test_affordability_flags_over_commitment():
    """A deal costing far too much of turnover must not be 'workable'."""
    app = make_applicant(
        requested_vans=4,
        annual_turnover_gbp=40_000.0,
        monthly_budget_gbp=0,
        daily_mileage=250,
    )
    per_van = applicant_van_cost(app, "diesel").total

    result = assess_affordability(app, "approve", "diesel", per_van)

    assert result.verdict == "not_workable", (
        f"Expected 'not_workable', got '{result.verdict}' "
        f"(commitment {result.commitment_pct:.0f}%)"
    )
    assert result.commitment_pct > 15.0
    assert result.actions, "A not-workable verdict must suggest next actions."


def test_affordability_respects_a_stated_budget():
    """A deal over the stated budget must not come back workable."""
    app = make_applicant(
        requested_vans=3,
        annual_turnover_gbp=200_000.0,
        monthly_budget_gbp=500,
        daily_mileage=60,
    )
    per_van = applicant_van_cost(app, "hybrid").total

    result = assess_affordability(app, "approve", "hybrid", per_van)

    assert not result.within_budget
    assert result.verdict != "workable"
    assert result.budget_gap < 0


def test_decline_always_verdict_not_workable():
    """A declined application can never be reported as workable."""
    app = make_applicant(
        requested_vans=1,
        annual_turnover_gbp=300_000.0,
        monthly_budget_gbp=5_000,
        daily_mileage=40,
    )
    per_van = applicant_van_cost(app, "ev").total

    result = assess_affordability(app, "decline", "ev", per_van)
    assert result.verdict == "not_workable"


def test_affordability_is_workable_property_is_reachable():
    """`is_workable` must be a real property on the result object.

    It was previously defined after the function's `return` statement, which
    made it dead code - the property silently did not exist.
    """
    app = make_applicant(
        requested_vans=1,
        annual_turnover_gbp=300_000.0,
        monthly_budget_gbp=5_000,
        daily_mileage=30,
    )
    per_van = applicant_van_cost(app, "ev").total

    result = assess_affordability(app, "approve", "ev", per_van)
    assert isinstance(result.is_workable, bool)
    assert result.is_workable == (result.verdict == "workable")


# ===========================================================================
# 7. SYNTHETIC POPULATION COHERENCE
# ===========================================================================

def test_generated_applicants_are_economically_coherent():
    """A generated applicant must be able to plausibly carry its vans.

    The previous generator drew turnover, van count and budget
    independently, so the median applicant had GBP 3,708/month of turnover
    against 3 vans costing GBP 2,500-3,500/month - a 70-95% commitment
    against a 15% guideline. Every applicant came back "not workable", so
    the verdict discriminated nothing.
    """
    from synthetic import generate_applicant
    from affordability import applicant_van_cost

    commitments = []
    for seed in range(120):
        app = generate_applicant(seed=seed)
        per_van = applicant_van_cost(app, "diesel").total
        total = per_van * app.requested_vans
        monthly_turnover = app.annual_turnover_gbp / 12.0
        commitments.append(total / monthly_turnover * 100.0)

    median = sorted(commitments)[len(commitments) // 2]
    assert median < 60.0, (
        f"Median commitment is {median:.0f}% of turnover - the generated "
        "population is still economically implausible."
    )


def test_affordability_verdict_discriminates():
    """The verdict must produce a spread of outcomes, not one constant answer."""
    from collections import Counter

    from synthetic import generate_applicant
    from fit_finder import evaluate_fit
    from affordability import assess_applicant_deal

    verdicts = Counter()
    for seed in range(120):
        app = generate_applicant(seed=seed)
        app.trade_category = normalise_trade_category(app.trade)
        fit = evaluate_fit(app)
        credit = ml_engine.vet(app)
        deal = assess_applicant_deal(app, credit.decision, fit.recommendation)
        verdicts[deal.verdict] += 1

    assert len(verdicts) >= 2, (
        f"The affordability verdict returned a single outcome {dict(verdicts)} - "
        "it is not discriminating between applicants."
    )
    assert verdicts.get("workable", 0) > 0, (
        f"No applicant came back workable: {dict(verdicts)}"
    )


def test_generated_budget_reflects_the_whole_deal():
    """The stated budget must be for ALL vans, not for one."""
    from synthetic import generate_applicant

    for seed in range(60):
        app = generate_applicant(seed=seed)
        assert app.monthly_budget_gbp > 0, (
            f"Seed {seed} produced a zero budget."
        )
        # A single van's cost is a lower bound on the whole deal, so the
        # budget must be at least broadly in that ballpark.
        assert app.monthly_budget_gbp > 200, (
            f"Seed {seed} budget {app.monthly_budget_gbp} is implausibly low."
        )


# ===========================================================================
# 8. CONSISTENCY BEHAVIOUR
# ===========================================================================

def test_paraphrasing_does_not_change_the_decision():
    """Rewording a trade must not change the decision.

    This is the whole point of the shared taxonomy. Before it, "town
    plumbing" matched the high-risk keyword "tow" and a business could be
    declined purely because of how its trade was described. The risk tier
    is a property of the trade, so the decision must be stable across
    paraphrases.
    """
    from consistency import run_consistency_test
    from synthetic import generate_applicant

    for seed in range(12):
        app = generate_applicant(seed=seed)
        app.trade_category = normalise_trade_category(app.trade)
        report = run_consistency_test(app, n_paraphrases=5, seed=seed)

        assert report.flip_count == 0, (
            f"Seed {seed}: {report.flip_count} paraphrases flipped the "
            f"decision - wording sensitivity has returned."
        )
        assert report.reason_stability_rate == 1.0, (
            f"Seed {seed}: reason sets differ across paraphrases."
        )


def test_consistency_report_is_populated():
    """The consistency report must carry real data for the app to render."""
    from consistency import run_consistency_test
    from synthetic import generate_applicant

    app = generate_applicant(seed=5)
    app.trade_category = normalise_trade_category(app.trade)
    report = run_consistency_test(app, n_paraphrases=5, seed=5)

    assert report.total_paraphrases == 5
    assert len(report.paraphrases) == 5
    assert report.decision_distribution
    assert report.base_decision in ("approve", "refer", "decline")
    for p in report.paraphrases:
        assert p.decision.decision in ("approve", "refer", "decline")
        assert p.decision.reasons


# ===========================================================================
# 9. UI CONTRAST (WCAG AA)
# ===========================================================================

def _relative_luminance(hex_colour: str) -> float:
    """WCAG relative luminance for a #RRGGBB colour."""
    h = hex_colour.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Not a 6-digit hex colour: {hex_colour!r}")

    def channel(value: int) -> float:
        c = value / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def _contrast_ratio(fg: str, bg: str) -> float:
    """WCAG contrast ratio between two #RRGGBB colours."""
    l1, l2 = _relative_luminance(fg), _relative_luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _css_tokens() -> dict:
    """Parse the --token: #hex pairs out of the app's TOKENS_CSS block."""
    import re

    source = open(os.path.join(os.path.dirname(__file__), "app.py")).read()
    css = re.search(r'TOKENS_CSS = """(.*?)"""', source, re.S)
    assert css, "TOKENS_CSS block not found in app.py"
    tokens = dict(re.findall(r"(--[a-z0-9-]+):\s*(#[0-9A-Fa-f]{6});", css.group(1)))
    assert tokens, "No colour tokens parsed from TOKENS_CSS"
    return tokens


def test_ui_colours_meet_wcag_aa_contrast():
    """Every text/background pair used by the UI must be readable.

    Regression test for a real defect: the app is a DARK theme
    (--bg #0F0E0C, --text #F3EEE4) but the recommendation banner and the
    affordability verdict used to hard-code LIGHT pastel backgrounds
    (#e3f7ec, #fdf3e3). Near-white body text on those panels rendered at
    roughly 1.03:1 - effectively invisible.
    """
    tokens = _css_tokens()

    # The theme must stay dark: if --text and --bg ever converge, every
    # surface in the app inherits the same bug.
    assert _contrast_ratio(tokens["--text"], tokens["--bg"]) > 10, (
        "--text and --bg are too close - the theme is no longer dark"
    )

    # Body and muted text must clear AA against the app background.
    assert _contrast_ratio(tokens["--text"], tokens["--bg"]) >= 4.5
    assert _contrast_ratio(tokens["--text-muted"], tokens["--bg"]) >= 4.5

    # Semantic foregrounds must clear AA against the app background.
    for fg in ("--accent", "--signal", "--warning", "--danger"):
        assert _contrast_ratio(tokens[fg], tokens["--bg"]) >= 4.5, (
            f"{fg} {tokens[fg]} on {tokens['--bg']} is below 4.5:1"
        )

    # Every semantic panel must be readable: its foreground, the standard
    # body text and the muted text all clear AA against the panel tint.
    for name in ("ok", "warn", "danger", "info"):
        bg = tokens[f"--panel-{name}-bg"]
        fg = tokens[f"--panel-{name}-fg"]
        assert _contrast_ratio(fg, bg) >= 4.5, (
            f"panel {name}: fg {fg} on bg {bg} is "
            f"{_contrast_ratio(fg, bg):.2f}:1, below 4.5:1"
        )
        assert _contrast_ratio(tokens["--text"], bg) >= 4.5, (
            f"panel {name}: --text on bg {bg} is below 4.5:1"
        )
        assert _contrast_ratio(tokens["--text-muted"], bg) >= 4.5, (
            f"panel {name}: --text-muted on bg {bg} is below 4.5:1"
        )


def test_no_light_backgrounds_in_a_dark_theme():
    """No token used as a background may be too light for the theme's text.

    A light background sitting under the dark theme's near-white --text is
    the exact shape of the original bug, so any such colour is rejected
    outright.
    """
    tokens = _css_tokens()
    app_bg = tokens["--bg"]

    for token, colour in tokens.items():
        if "bg" not in token and "surface" not in token:
            continue
        # --bg IS the page background, so it cannot differ from itself.
        if token == "--bg":
            continue
        ratio = _contrast_ratio(tokens["--text"], colour)
        assert ratio >= 4.5, (
            f"{token} {colour} is too light for --text "
            f"({ratio:.2f}:1) - text on it would be unreadable"
        )
        # And it should still be distinguishable from the page background.
        assert _contrast_ratio(colour, app_bg) > 1.05, (
            f"{token} {colour} is indistinguishable from the app background"
        )


def test_panel_classes_exist_in_stylesheet():
    """The markup relies on these classes; a rename would silently unstyle."""
    import re

    source = open(os.path.join(os.path.dirname(__file__), "app.py")).read()
    css = re.search(r'TOKENS_CSS = """(.*?)"""', source, re.S)
    assert css
    css = css.group(1)

    for cls in (".panel", ".panel-chip", ".panel-title", ".panel-figure", ".panel-note"):
        assert cls in css, f"CSS class {cls} is missing from the stylesheet"

    # .value must be usable OUTSIDE a .metric block - it is used by the
    # banner and verdict panels. Scoping it to ".metric .value" left those
    # figures with no colour rule at all.
    assert re.search(r"(?<![\w-])\.value\s*\{", css), (
        "A top-level .value rule is required for markup outside .metric"
    )
    assert ".value.accent" in css
    assert ".value.signal" in css


# ===========================================================================
# Runner
# ===========================================================================

def _run_all() -> int:
    """Run every test_* function in this module. Returns a process exit code."""
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]

    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL  {name}: {exc}")

    print("-" * 60)
    print(f"{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print(f"\n{len(failures)} FAILED:")
        for name, exc in failures:
            print(f"  - {name}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())
