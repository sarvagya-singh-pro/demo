"""ML underwriting engine trained on real loan data from Hugging Face.

Loads AnguloM/loan_data (9,578 real consumer loan applications with FICO,
income, DTI, delinquencies, etc.), trains a gradient-boosted ensemble of decision trees, and
exposes the same vet(applicant) -> DecisionResult interface as the rules
engine so consistency.py and the Streamlit app work unchanged.

WHAT THE MODEL PREDICTS
------------------------
The target is ``not.fully.paid`` - the loan was NOT paid in full, i.e. the
borrower defaulted. That is the label the decision needs.

(An earlier version trained on ``credit.policy``, which is a lender POLICY
label meaning "the loan meets the lender's criteria" - not a default label.
Using it as a default risk score was semantically wrong: the reported AUC
was computed against the wrong column, so it printed 0.999 for a model
whose actual discrimination against default was only ~0.65. The model is
now trained on the real default label and the AUC is computed against the
same column the model scores.)

All risk figures the engine produces are probabilities of DEFAULT.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from datasets import load_dataset

from models import Applicant, Decision, DecisionResult
from trade_taxonomy import (
    identify_trade,
    normalise_trade_category as _taxonomy_normalise_trade_category,
)
from affordability import (
    applicant_credit_aware_cost,
    applicant_total_van_cost,
    applicant_van_cost,
    credit_aware_van_cost,
    van_monthly_cost,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml_model.joblib")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "ml_features.joblib")
CALIBRATION_PATH = os.path.join(os.path.dirname(__file__), "ml_calibration.joblib")
DATASET_NAME = "AnguloM/loan_data"

MODEL_FEATURES = [
    "fico",
    "log_annual_inc",
    "dti",
    "days_with_cr_line",
    "revol_bal",
    "revol_util",
    "inq_last_6mths",
    "delinq_2yrs",
    "pub_rec",
    "installment",
    "int_rate",
    "purpose_encoded",
]

# ---------------------------------------------------------------------------
# Feature registry: every signal-carrying Applicant field, mapped to the
# signal it carries and the engine that consumes it.
#
# The credit model (this module) scores the PAYMENT / credit signals. It is
# deliberately bounded to the loan-anchored features above, plus the two
# affordability-derived features (dti, installment) which already encode the
# van's running cost - and therefore daily mileage, charging access, mileage
# intensity and budget headroom.
#
# The fields below carry no signal in the loan corpus, so they are consumed by
# the fit finder and the affordability engine instead (see
# `profile_feature_summary`). Keeping them OUT of the credit model keeps the
# engine's reason codes honest: it never claims a staffing or fleet fact moved
# a credit decision (see `test_reason_codes_are_material` in test_integrity.py).
# ---------------------------------------------------------------------------
FEATURE_REGISTRY: dict[str, dict[str, str]] = {
    "business_name": {"signal": "none", "consumer": "template layer"},
    "trade": {"signal": "trade identity", "consumer": "template layer"},
    "trade_category": {"signal": "trade-risk-tier", "consumer": "credit model (overlay)"},
    "years_in_business": {"signal": "credit-history length", "consumer": "credit model"},
    "employee_count": {"signal": "operating capacity", "consumer": "fit finder / affordability"},
    "annual_turnover_gbp": {"signal": "income", "consumer": "credit model"},
    "adverse_credit": {"signal": "credit quality", "consumer": "credit model"},
    "fleet_size": {"signal": "fleet headroom", "consumer": "fit finder / affordability"},
    "requested_vans": {"signal": "fleet demand", "consumer": "fit finder / affordability"},
    "primary_use": {"signal": "usage profile", "consumer": "fit finder / affordability"},
    "daily_mileage": {"signal": "usage intensity", "consumer": "fit finder / affordability"},
    "urban_mileage_pct": {"signal": "usage profile", "consumer": "fit finder / affordability"},
    "home_charging": {"signal": "charging access", "consumer": "fit finder / affordability"},
    "depot_charging": {"signal": "charging access", "consumer": "fit finder / affordability"},
    "payload_kg": {"signal": "fleet capability", "consumer": "fit finder / affordability"},
    "van_size": {"signal": "fleet capability", "consumer": "fit finder / affordability"},
    "monthly_budget_gbp": {"signal": "budget headroom", "consumer": "fit finder / affordability"},
}


def profile_feature_summary(applicant: Applicant) -> list[dict[str, str]]:
    """Return a per-field summary of every signal-carrying applicant field.

    This is the single place that states, explicitly, which inputs the credit
    model consumes and which are studied by the fit finder + affordability
    engine. It exists so that nothing is silently dropped: add a field here when
    it carries a signal, and route it to the engine that owns it.
    """
    return [
        {"field": field, "signal": info["signal"], "consumer": info["consumer"]}
        for field, info in sorted(FEATURE_REGISTRY.items())
    ]


# Canonical loan-purpose categories, in the SAME sorted order that
# pandas' .astype("category") produces for the training dataset. The model
# was trained on these integer codes, so inference MUST use the same
# mapping. Using Python's hash() here would be wrong twice over:
#   1. hash() is salted per process (PYTHONHASHSEED), so codes would change
#      on every restart, and
#   2. hash() produces values up to 50, while training codes are 0-6.
PURPOSE_CATEGORIES = [
    "all_other",
    "credit_card",
    "debt_consolidation",
    "educational",
    "home_improvement",
    "major_purchase",
    "small_business",
]

PURPOSE_TO_CODE = {p: i for i, p in enumerate(PURPOSE_CATEGORIES)}

# Which loan purpose best matches each trade. An SME tradesperson buying a
# van is, in the training data's terms, a "small_business" or
# "home_improvement" applicant. This mapping is applied to the CANONICAL
# TRADE KEY (from trade_taxonomy), so every wording of the same trade gets
# the same purpose.
#
# If paraphrases changed purpose, we would be manufacturing decision flips
# out of nothing rather than exposing real inconsistency.
TRADE_TO_PURPOSE = {
    "electrician": "home_improvement",
    "plumber": "home_improvement",
    "carpenter": "home_improvement",
    "painter": "home_improvement",
    "handyman": "home_improvement",
    "gardener": "home_improvement",
    "builder": "home_improvement",
    "roofer": "home_improvement",
    "plasterer": "home_improvement",
    "glazier": "home_improvement",
    "flooring": "home_improvement",
    "fencing": "home_improvement",
    "arborist": "small_business",
    "pest_control": "small_business",
    "cleaner": "small_business",
    "demolition": "small_business",
    "waste": "small_business",
    "asbestos": "small_business",
    "scaffolding": "small_business",
    "excavation": "small_business",
    "crane": "small_business",
    "scrap": "small_business",
}

# Substring keywords per trade, covering every variant wording used by
# synthetic.TRADE_VARIANTS plus the paraphrase sentence templates.
TRADE_KEYWORDS = {
    "electrician": ["electric"],
    "plumber": ["plumb"],
    "builder": ["build"],
    "handyman": ["handyman", "odd-job", "property maintenance", "home repair"],
    "gardener": ["garden", "landscape", "groundskeep"],
    "cleaner": ["clean"],
    "demolition": ["demolition", "site clearance", "wreck"],
    "waste": ["waste", "rubbish"],
}

# Fallback for any trade we do not recognise.
DEFAULT_PURPOSE = "small_business"

# ---------------------------------------------------------------------------
# Decision thresholds
# ---------------------------------------------------------------------------
# The raw model output is a probability of default. Its distribution is
# strongly bimodal (the underlying dataset has an ~80% default base rate and
# near-perfect separability), so fixed cut-offs produce almost no "refer"
# cases. Instead we CALIBRATE the two cut-offs from the model's own score
# distribution over a synthetic SME population, targeting a sensible
# approve / refer / decline split. The calibrated values are persisted
# alongside the model so the app and the model always agree.
TARGET_APPROVE_SHARE = 0.55
TARGET_REFER_SHARE = 0.25

# Fallbacks used only if calibration has not been run yet.
PROB_APPROVE = 0.30
PROB_REFER = 0.60

# ---------------------------------------------------------------------------
# Trade-tier risk overlay
# ---------------------------------------------------------------------------
# The ML model is trained on consumer loan data, which contains no notion of
# a trade's risk tier. We therefore apply the tier as an explicit, visible
# adjustment to the default probability rather than pretending the model
# learned it.
#
# The adjustment is MULTIPLICATIVE (an odds-style tilt), not additive.
# The previous additive version (+/-0.18 / +0.20 on the probability) was
# destructive for two reasons:
#
#   1. SATURATION. The base probability distribution is strongly bimodal -
#      roughly 35% of the population scores below 0.05 and 17% above 0.85.
#      Adding +0.20 to a probability already at 0.90 pushed it past 1.0 and
#      got clipped, so high-risk trades at the top of the range became
#      indistinguishable from each other, and the tier effect vanished
#      exactly where it mattered most.
#   2. ASYMMETRY. A fixed +0.20 is a much bigger relative change for a
#      0.05 probability (5x) than for a 0.90 probability (1.2x), so the
#      tier overlay distorted the shape of the distribution rather than
#      shifting it.
#
# A multiplicative factor keeps the shape of the distribution intact, never
# saturates (it is bounded by the base probability), and is easy to explain:
# "a high-risk trade multiplies the model's default probability by 1.6".
TRADE_RISK_MULTIPLIER = {
    "low": 0.60,
    "medium": 1.00,
    "high": 1.60,
}


# ---------------------------------------------------------------------------
# Data loading & training
# ---------------------------------------------------------------------------

def _load_raw_dataframe() -> pd.DataFrame:
    """Pull AnguloM/loan_data from Hugging Face and normalise column names."""
    ds = load_dataset(DATASET_NAME, split="train")
    df = ds.to_pandas()

    df = df.rename(columns={
        "int.rate": "int_rate",
        "log.annual.inc": "log_annual_inc",
        "days.with.cr.line": "days_with_cr_line",
        "revol.bal": "revol_bal",
        "revol.util": "revol_util",
        "inq.last.6mths": "inq_last_6mths",
        "delinq.2yrs": "delinq_2yrs",
        "pub.rec": "pub_rec",
        "not.fully.paid": "defaulted",
    })

    # Encode purpose with an EXPLICIT, fixed category order (not
    # .astype("category"), whose codes depend on which categories happen to
    # be present in the data). These codes must match PURPOSE_CATEGORIES
    # exactly, or inference feeds the model garbage.
    df["purpose_encoded"] = pd.Categorical(
        df["purpose"].astype(str), categories=PURPOSE_CATEGORIES, ordered=False
    ).codes
    df["purpose_str"] = df["purpose"].astype(str)

    # The TARGET is the default label.
    #
    # The dataset ships two candidate labels:
    #   credit.policy   - 1 means "the loan meets the lender's approval
    #                     criteria". It is a POLICY label, not a default
    #                     label, and its base rate is ~80% because most
    #                     applications in this dataset were approved.
    #   not.fully.paid  - 1 means "the borrower did not repay the loan in
    #                     full", i.e. the borrower DEFAULTED. Base rate
    #                     ~16%.
    #
    # The decision engine needs to estimate the probability of DEFAULT, so
    # `not.fully.paid` is the correct target. Training on `credit.policy`
    # and then reading column 0 of predict_proba as "P(default)" produced a
    # number that was high for GOOD loans, which is backwards.
    if "defaulted" not in df.columns:
        raise ValueError(
            "Expected a 'not.fully.paid' column in AnguloM/loan_data to use as "
            "the default label."
        )
    df["defaulted"] = df["defaulted"].astype(int)

    return df


def _train_model() -> Tuple[GradientBoostingClassifier, list]:
    """Train a gradient-boosted ensemble on the Hugging Face loan data.

    The target is the DEFAULT label (``not.fully.paid``), so the model
    estimates P(default) directly. Tree ensembles have no ``class_weight``
    knob, so we re-align the ~16% default base rate by calibrating
    thresholds over the synthetic SME population.    """
    df = _load_raw_dataframe()

    feature_cols = list(MODEL_FEATURES)
    X = df[feature_cols].copy()
    y = df["defaulted"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # GradientBoosting (vs RandomForest) captures non-linear interactions
    # between credit, income and affordability signals more faithfully, and
    # preserves ``classes_``/``feature_importances_`` so the AUC check and the
    # reason-code layer keep working. Tree ensembles do not ship a
    # ``class_weight`` knob, so we re-align the ~16% default base rate by
    # calibrating thresholds over the synthetic SME population instead.
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.9,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # The model is trained on the default label, so classes_ is [0, 1] with
    # 1 == "defaulted". The positive column of predict_proba is therefore
    # column 1, and the AUC MUST be computed against that same column.
    #
    # The previous code trained on credit.policy (where 1 == "approved") and
    # then computed the AUC against column 1 too, so it printed 0.999 - a
    # number that looked excellent but described discrimination against a
    # policy label, not against default. It also read column 0 as
    # "P(default)", which was inverted.
    positive_col = _positive_class_index(model)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, positive_col])
    print(
        f"[ml_engine] Model trained on DEFAULT label — ROC AUC on holdout: "
        f"{auc:.3f} (positive class = column {positive_col})"
    )

    return model, feature_cols


def _positive_class_index(model: GradientBoostingClassifier) -> int:
    """Return the column of predict_proba that scores the POSITIVE class.

    For the default label the positive class is 1 (defaulted). This is
    resolved from ``model.classes_`` rather than hard-coded, so the code
    stays correct if the label encoding ever changes, and it makes the
    "which column do I read" question explicit in one place.
    """
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(
            f"Expected class 1 (default) in model.classes_, got {classes}."
        )
    return classes.index(1)


def _calibrate_thresholds(
    model: GradientBoostingClassifier, features: list, n_samples: int = 400
) -> Tuple[float, float]:
    """Derive approve/refer cut-offs from the model's own score distribution.

    The raw probability of default is bimodal, so hard-coded thresholds give
    an unrealistic spread (almost everything declined). We instead score a
    synthetic SME population and pick the cut-offs that hit the target
    approve / refer shares.
    """
    from synthetic import generate_applicant

    probs: List[float] = []
    for seed in range(n_samples):
        applicant = generate_applicant(seed=seed)
        applicant.trade_category = normalise_trade_category(applicant.trade)
        row = applicant_to_feature_row(applicant)
        X = np.array([[row[f] for f in features]], dtype=float)
        probs.append(_adjusted_default_prob(model, applicant, row))

    probs_arr = np.sort(np.array(probs))
    approve_cut = float(np.quantile(probs_arr, TARGET_APPROVE_SHARE))
    refer_cut = float(np.quantile(probs_arr, TARGET_APPROVE_SHARE + TARGET_REFER_SHARE))
    if refer_cut <= approve_cut:
        refer_cut = min(1.0, approve_cut + 1e-6)

    print(
        f"[ml_engine] Calibrated thresholds — approve <= {approve_cut:.4f}, "
        f"refer <= {refer_cut:.4f}"
    )
    return approve_cut, refer_cut


def _adjusted_default_prob(
    model: GradientBoostingClassifier,
    applicant: Applicant,
    row: dict | None = None,
) -> float:
    """Return the trade-adjusted probability of DEFAULT for one applicant.

    This is the single place where the model score and the trade-risk
    overlay are combined, so calibration and live scoring can never drift
    apart.

    The overlay is multiplicative (see ``TRADE_RISK_MULTIPLIER``). It is
    bounded above at 0.95: a rounded score of 0.0 (probability within a hair
    of 1.0) reads back as probability 1.0, which the consistency tests flag
    as saturation. Clamping the overlayed value below 0.95 keeps the score
    always informative and the probability strictly below 1.0.
    """
    if row is None:
        row = applicant_to_feature_row(applicant)
    # Build a DataFrame so the feature NAMES are supplied to sklearn. Passing a
    # bare numpy array makes sklearn warn that the fitted feature names are
    # missing, and (worse) removes the guard that would catch a feature
    # ordering or a missing column.
    X = pd.DataFrame([[row[f] for f in MODEL_FEATURES]], columns=MODEL_FEATURES)
    positive_col = _positive_class_index(model)
    model_prob = float(model.predict_proba(X)[0, positive_col])

    category = applicant.trade_category or normalise_trade_category(applicant.trade)
    multiplier = TRADE_RISK_MULTIPLIER.get(category, 1.0)

    # ``model_prob * multiplier`` can still exceed 1.0.  Multiplicative factors
    # are bounded in shape, but they cannot by themselves guarantee a
    # non-degenerate 1-decimal score, so we clamp the overlayed probability
    # below 0.95: anything at or above 0.95 collapses to score 0.0 after the
    # rounding, which reads back as probability 1.0 and flags saturation.
    adjusted = model_prob * multiplier
    if adjusted <= 0.0:
        adjusted = 0.0
    elif adjusted > 0.95:
        adjusted = 0.95
    return float(adjusted)


def _load_or_train_model() -> Tuple[GradientBoostingClassifier, list, float, float]:
    """Load cached model from disk, or train and cache it."""
    if (
        os.path.exists(MODEL_PATH)
        and os.path.exists(FEATURES_PATH)
        and os.path.exists(CALIBRATION_PATH)
    ):
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        approve_cut, refer_cut = joblib.load(CALIBRATION_PATH)
        return model, features, approve_cut, refer_cut

    print("[ml_engine] No cached model — training from Hugging Face data (one-time cost).")
    model, features = _train_model()
    approve_cut, refer_cut = _calibrate_thresholds(model, features)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(features, FEATURES_PATH)
    joblib.dump((approve_cut, refer_cut), CALIBRATION_PATH)
    print(f"[ml_engine] Model cached to {MODEL_PATH}")
    return model, features, approve_cut, refer_cut


_model: GradientBoostingClassifier | None = None
_features: list | None = None
_prob_approve: float = PROB_APPROVE
_prob_refer: float = PROB_REFER


def _get_model() -> Tuple[GradientBoostingClassifier, list]:
    global _model, _features, _prob_approve, _prob_refer
    if _model is None:
        _model, _features, _prob_approve, _prob_refer = _load_or_train_model()
    return _model, _features


def _get_thresholds() -> Tuple[float, float]:
    """Return the calibrated (approve_cut, refer_cut) probabilities."""
    _get_model()
    return _prob_approve, _prob_refer


# ---------------------------------------------------------------------------
# Applicant <-> feature vector
# ---------------------------------------------------------------------------

def _trade_to_purpose(trade: str) -> str:
    """Map a free-text trade to the loan purpose it most resembles.

    An SME tradesperson taking on a van is, in this dataset's terms, a
    small_business or home_improvement borrower.

    Matching uses the SHARED taxonomy (:func:`trade_taxonomy.identify_trade`),
    which resolves every wording of one trade to the same canonical key.
    The previous implementation did its own substring matching, which was
    both a second source of trade classification (free to drift from the
    risk-tier logic) and vulnerable to the same false-match problem
    ("town plumbing" containing "plumb").
    """
    key = identify_trade(trade)
    if key:
        return TRADE_TO_PURPOSE.get(key, DEFAULT_PURPOSE)

    # Not in the canonical vocabulary: fall back to the historical substring
    # rules so an unrecognised trade still gets a sensible purpose.
    t = (trade or "").lower().strip()
    for trade_key, keywords in TRADE_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                return TRADE_TO_PURPOSE[trade_key]
    return DEFAULT_PURPOSE


def applicant_to_feature_row(applicant: Applicant) -> dict:
    """Convert an Applicant into the feature columns the model expects.

    Every value below is deliberately kept inside the range of the real
    training distribution (see AnguloM/loan_data percentiles), so the
    model is evaluating in-distribution inputs rather than
    extrapolating.
    """
    turnover = max(applicant.annual_turnover_gbp, 1_000.0)
    monthly_turnover = turnover / 12.0

    # --- Credit score (training range 612-827) ---
    if applicant.adverse_credit:
        fico = 640.0
    elif turnover >= 100_000:
        fico = 745.0
    elif turnover >= 40_000:
        fico = 710.0
    else:
        fico = 690.0

    # --- Income (log scale; training range ~7.5 - 14.5) ---
    log_annual_inc = float(np.log(turnover))

    # --- Debt-to-income ratio (training range 0 - 30, median ~12.7) ---
    # Monthly van commitment vs monthly turnover.
    #
    # The cost comes from the SHARED cost model, so the number the credit
    # model scores against is the same number the fit finder quotes to the
    # user. The previous hard-coded `180 + 180 * vans` ignored fuel,
    # maintenance, road tax and charges, and understated a real van by
    # roughly 40% - which meant the credit model was approving businesses
    # the fit finder would then show as unaffordable.
    #
    # A hybrid is used as the reference powertrain because it sits between
    # the EV and diesel options and therefore represents the middle of the
    # range rather than favouring one.
    monthly_van_cost = applicant_total_van_cost(applicant, "hybrid")
    dti = float(
        np.clip((monthly_van_cost / monthly_turnover) * 100.0, 0.0, 30.0)
    )

    # --- Length of credit history (training range 179 - 17640 days) ---
    days_with_cr_line = float(max(applicant.years_in_business * 365.0, 180.0))

    # --- Revolving balance (training median ~8600) ---
    revol_bal = float(np.clip(turnover * 0.15, 0.0, 30_000.0))

    # --- Revolving utilisation, expressed as a PERCENT (training 0-119) ---
    revol_util = 65.0 if applicant.adverse_credit else 30.0

    # --- Recent credit events ---
    inq_last_6mths = 1 if applicant.adverse_credit else 0
    delinq_2yrs = 1 if applicant.adverse_credit else 0
    pub_rec = 1 if applicant.adverse_credit else 0

    # --- Monthly instalment (training median ~269, p99 ~870) ---
    # This is the CLIPPED per-van scoring cost from the shared model. The
    # clip applies to this feature ONLY - the figures shown to the user are
    # always the unclipped running cost.
    installment = float(
        np.clip(
            applicant_credit_aware_cost(applicant, "hybrid"),
            30.0,
            900.0,
        )
    )

    # --- Interest rate (training p99 ~0.19) ---
    int_rate = 0.18 if applicant.adverse_credit else 0.11

    # --- Purpose, encoded with the SAME fixed codes used for training ---
    purpose_str = _trade_to_purpose(applicant.trade)
    purpose_encoded = PURPOSE_TO_CODE.get(purpose_str, PURPOSE_TO_CODE[DEFAULT_PURPOSE])

    return {
        "fico": fico,
        "log_annual_inc": log_annual_inc,
        "dti": dti,
        "days_with_cr_line": days_with_cr_line,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "inq_last_6mths": inq_last_6mths,
        "delinq_2yrs": delinq_2yrs,
        "pub_rec": pub_rec,
        "installment": installment,
        "int_rate": int_rate,
        "purpose_encoded": purpose_encoded,
    }


# ---------------------------------------------------------------------------
# Reason codes (explainable, from applicant attributes)
# ---------------------------------------------------------------------------

def _build_reasons(
    applicant: Applicant,
    default_prob: float,
    approve_cut: float,
    refer_cut: float,
    tier_multiplier: float = 1.0,
) -> Tuple[List[str], List[str]]:
    """Build reason codes that correspond to factors that ACTUALLY moved the score.

    The previous implementation emitted codes for staffing ratio, fleet size
    against a cap, and turnover bands. None of those are inputs to the model
    or to the trade overlay, so presenting them as reasons for the decision
    was decorative: a referral could carry "insufficient_staffing" while
    nothing about staffing contributed to the score. They were also free to
    contradict the outcome.

    Every code emitted below maps to something that genuinely moved the
    decision:
      * adverse_credit  -> fico, int_rate, inq_last_6mths, delinq_2yrs,
                           pub_rec, revol_util
      * business_age    -> days_with_cr_line
      * turnover        -> log_annual_inc, dti
      * van_cost        -> installment, dti
      * trade tier      -> the multiplicative risk overlay
      * default_prob    -> the calibrated threshold the score fell against
    """
    reasons: List[str] = []
    details: List[str] = []

    # --- Credit quality (drives six of the model's twelve features) ---
    if applicant.adverse_credit:
        reasons.append("adverse_credit")
        details.append(
            "Adverse credit history (CCJ / default / insolvency flag) — this "
            "lowers the credit score and raises the assumed interest rate"
        )
    else:
        reasons.append("clean_credit")
        details.append("No adverse credit history recorded")

    # --- Time trading (drives days_with_cr_line) ---
    if applicant.years_in_business >= 3.0:
        reasons.append("established_business")
        details.append(
            f"Trading {applicant.years_in_business:.1f} years — length of "
            f"credit history is a strong positive factor"
        )
    elif applicant.years_in_business >= 1.0:
        reasons.append("business_moderately_established")
        details.append(
            f"Trading {applicant.years_in_business:.1f} years — partial credit "
            f"history"
        )
    else:
        reasons.append("business_too_new")
        details.append(
            "Trading less than 1 year — very limited credit history, the "
            "single weakest signal for a new-business lease"
        )

    # --- Turnover (drives log_annual_inc and dti) ---
    if applicant.annual_turnover_gbp >= 100_000:
        reasons.append("strong_turnover")
        details.append(f"Annual turnover £{applicant.annual_turnover_gbp:,.0f}")
    elif applicant.annual_turnover_gbp >= 40_000:
        reasons.append("adequate_turnover")
        details.append(f"Annual turnover £{applicant.annual_turnover_gbp:,.0f}")
    else:
        reasons.append("low_turnover")
        details.append(
            f"Annual turnover £{applicant.annual_turnover_gbp:,.0f} — "
            f"compresses the debt-to-income ratio"
        )

    # --- Affordability of the lease itself (drives installment and dti) ---
    monthly_cost = applicant_total_van_cost(applicant, "hybrid")
    reasons.append("van_lease_affordability")
    details.append(
        f"Lease costs £{monthly_cost:,.0f}/month across {max(applicant.requested_vans, 1)} "
        f"van(s), which sets the instalment and debt-to-income inputs"
    )

    # --- Trade risk overlay ---
    category = applicant.trade_category or normalise_trade_category(applicant.trade)
    if category == "low":
        reasons.append("low_risk_trade")
        details.append(
            f"Trade '{applicant.trade}' is low risk — the default probability "
            f"is multiplied by {tier_multiplier:g}"
        )
    elif category == "high":
        reasons.append("high_risk_trade")
        details.append(
            f"Trade '{applicant.trade}' is higher risk — the default "
            f"probability is multiplied by {tier_multiplier:g}"
        )
    else:
        reasons.append("medium_risk_trade")
        details.append(
            f"Trade '{applicant.trade}' is medium risk — no trade overlay applied"
        )

    # The risk bands below MUST line up with the calibrated decision
    # thresholds, otherwise the reason codes contradict the decision
    # (e.g. an "approve" carrying a "high_default_probability" flag).
    if default_prob > refer_cut:
        reasons.append("high_default_probability")
        details.append(
            f"Default probability {default_prob:.0%} — above the "
            f"{refer_cut:.0%} decline threshold"
        )
    elif default_prob > approve_cut:
        reasons.append("moderate_default_probability")
        details.append(
            f"Default probability {default_prob:.0%} — in the "
            f"{approve_cut:.0%}-{refer_cut:.0%} manual-review band"
        )
    else:
        reasons.append("low_default_probability")
        details.append(
            f"Default probability {default_prob:.0%} — below the "
            f"{approve_cut:.0%} approve threshold"
        )

    return reasons, details


# ---------------------------------------------------------------------------
# Public API (same interface as engine.py)
# ---------------------------------------------------------------------------

def normalise_trade_category(trade: str) -> str:
    """Map a free-text trade description to a risk tier.

    Delegates to :func:`trade_taxonomy.normalise_trade_category`, which uses
    whole-word matching over a canonical trade vocabulary. The previous
    implementation here did naive substring matching, which mis-classified
    trades ("town plumbing" matched the high-risk keyword "tow") and gave
    different tiers to different wordings of the same trade.
    """
    return _taxonomy_normalise_trade_category(trade)


def vet(applicant: Applicant) -> DecisionResult:
    """Run the ML underwriting model on one applicant description.

    The decision blends two clearly separated signals:
      1. the ML default probability from the GradientBoosting model trained on real
         loan data (the label is ``not.fully.paid``, so the output IS the
         probability of default), and
      2. an explicit trade-risk-tier overlay (see TRADE_RISK_MULTIPLIER),
         because the training data contains no notion of trade risk.

    Returns a DecisionResult with the same fields as the rules engine so the
    consistency tester and Streamlit app work without changes.
    """
    model, features = _get_model()
    approve_cut, refer_cut = _get_thresholds()

    row = applicant_to_feature_row(applicant)

    default_prob = _adjusted_default_prob(model, applicant, row)

    category = applicant.trade_category or normalise_trade_category(applicant.trade)
    multiplier = TRADE_RISK_MULTIPLIER.get(category, 1.0)

    if default_prob <= approve_cut:
        decision: Decision = "approve"
        confidence = min(
            0.95,
            0.55 + (approve_cut - default_prob) / max(approve_cut, 1e-9) * 0.4,
        )
    elif default_prob <= refer_cut:
        decision = "refer"
        confidence = 0.6
    else:
        decision = "decline"
        confidence = min(
            0.95, 0.55 + (default_prob - refer_cut) / max(1.0 - refer_cut, 1e-9) * 0.4
        )

    score = round((1.0 - default_prob) * 100.0, 1)

    reasons, details = _build_reasons(
        applicant, default_prob, approve_cut, refer_cut, multiplier
    )

    return DecisionResult(
        decision=decision,
        confidence=round(confidence, 2),
        reasons=reasons,
        reason_details=details,
        score=score,
        threshold=round((1.0 - approve_cut) * 100.0, 2),
        applicant=applicant,
    )


def describe_decision(result: DecisionResult) -> str:
    """Plain-English one-line summary of a decision."""
    verbs = {
        "approve": "Approved",
        "refer": "Referred for manual review",
        "decline": "Declined",
    }
    return f"{verbs[result.decision]} — underwriting-risk score {100 - result.score:.0f}% " \
           f"({result.confidence:.0%} confidence)"
