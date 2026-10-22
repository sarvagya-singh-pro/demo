"""Underwriting consistency tester.

Runs the same underlying SME business through the vetting engine under several
different wordings (paraphrases) and measures how often the decision flips.
Framed as *vetting consistency*, never credit scoring.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from models import Applicant, ConsistencyReport, ParaphraseResult
from ml_engine import vet, normalise_trade_category


def run_consistency_test(
    applicant: Applicant, n_paraphrases: int = 5, seed: int = 7
) -> ConsistencyReport:
    """Vet one business under N paraphrases; aggregate flip stats."""

    from synthetic import generate_paraphrases, apply_paraphrase

    paraphrases = generate_paraphrases(applicant, n=n_paraphrases, seed=seed)

    results: List[ParaphraseResult] = []
    for label, trade_desc in paraphrases:
        app_variant = apply_paraphrase(applicant, trade_desc)
        # ensure category normalised from the (possibly new) wording
        app_variant.trade_category = normalise_trade_category(app_variant.trade)
        decision = vet(app_variant)
        results.append(
            ParaphraseResult(
                label=label,
                trade_description=trade_desc,
                decision=decision,
                flipped=False,  # filled below
            )
        )

    base = results[0]
    flipped_count = 0
    decision_distribution: dict = {}
    identical_reason_count = 0

    base_reasons = tuple(sorted(base.decision.reasons))
    for r in results:
        r.flipped = r.decision.decision != base.decision.decision
        if r.flipped:
            flipped_count += 1
        decision_distribution[r.decision.decision] = (
            decision_distribution.get(r.decision.decision, 0) + 1
        )
        if tuple(sorted(r.decision.reasons)) == base_reasons:
            identical_reason_count += 1

    total = max(len(results), 1)
    flip_rate = flipped_count / total
    reason_stability = identical_reason_count / total

    return ConsistencyReport(
        applicant=applicant,
        base_decision=base.decision.decision,
        base_reasons=base.decision.reasons,
        total_paraphrases=total,
        flip_count=flipped_count,
        flip_rate=round(flip_rate, 3),
        decision_distribution=decision_distribution,
        reason_stability_rate=round(reason_stability, 3),
        paraphrases=results,
    )


def batch_consistency_test(
    applicants: List[Applicant], n_paraphrases: int = 5, seed: int = 7
) -> List[ConsistencyReport]:
    """Run the consistency test across many applicants."""
    return [
        run_consistency_test(app, n_paraphrases=n_paraphrases, seed=seed + i)
        for i, app in enumerate(applicants)
    ]


def aggregate_flip_rate(reports: List[ConsistencyReport]) -> dict:
    """Roll-up statistics across a batch of consistency reports."""
    if not reports:
        return {
            "applicants_tested": 0,
            "total_paraphrases": 0,
            "overall_flip_rate": 0.0,
            "avg_reason_stability": 0.0,
            "flipped_count": 0,
            "decision_distribution": {},
        }
    total_paraphrases = sum(r.total_paraphrases for r in reports)
    total_flips = sum(r.flip_count for r in reports)
    avg_stability = sum(r.reason_stability_rate for r in reports) / len(reports)

    dist: dict = {}
    for r in reports:
        for d, c in r.decision_distribution.items():
            dist[d] = dist.get(d, 0) + c

    return {
        "applicants_tested": len(reports),
        "total_paraphrases": total_paraphrases,
        "overall_flip_rate": round(total_flips / max(total_paraphrases, 1), 3),
        "avg_reason_stability": round(avg_stability, 3),
        "flipped_count": total_flips,
        "decision_distribution": dist,
    }


def find_flip_prone_applicant(
    seed_range: range = range(0, 300), n_paraphrases: int = 5
) -> Tuple[Optional[Applicant], Optional[ConsistencyReport]]:
    """Scan synthetic applicants and return one whose decision flips across
    paraphrases, preferring cases that span approve <-> decline.

    Returns (applicant, ConsistencyReport) or (None, None) if none found.

    NOTE ON THE CURRENT RESULT
    --------------------------
    With the shared trade taxonomy in place, the risk tier is a property of
    the TRADE rather than of the sentence describing it, so paraphrasing a
    business's trade description no longer changes its tier - and therefore
    no longer changes its decision. The flip rate across the synthetic
    population is 0%.

    That is the CORRECT outcome: the original defect was a wording artefact
    masquerading as risk. This function is retained because it still has a
    job to do - it is the regression tripwire. If a future change to the
    taxonomy, the overlay or the model reintroduces wording sensitivity,
    this scan will find the affected applicant and it will show up as a
    non-zero flip rate on the Consistency page.
    """
    from synthetic import generate_applicant

    best = None
    for seed in seed_range:
        app = generate_applicant(seed=seed)
        app.trade_category = normalise_trade_category(app.trade)
        report = run_consistency_test(app, n_paraphrases=n_paraphrases, seed=seed)
        if report.flip_count == 0:
            continue
        decisions = {p.decision.decision for p in report.paraphrases}
        spans_broad = bool({"approve", "decline"} & decisions) and len(decisions) >= 2
        score = (report.flip_count, 1 if spans_broad else 0)
        if best is None or score > best[0]:
            best = (score, app, report)
            if spans_broad:
                break
    if best is None:
        return None, None
    return best[1], best[2]
