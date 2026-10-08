"""Judge reliability: agreement between the LLM judge and objective ground truth.

Stdlib only.

This module is what turns "I used an LLM as a judge" from an act of faith into a number. GSM8K has
an objectively known answer, so on those items we can score with *both* the deterministic grader
and the LLM judge, then measure how often the judge agrees with arithmetic truth (design §8.3).

Raw agreement alone is not enough, and the reason matters: if 90% of answers are correct, a judge
that blindly replies "correct" scores 90% agreement while being completely useless. Cohen's kappa
corrects for that chance agreement -- the always-correct judge scores kappa ~= 0.

The two error directions are also reported separately, because they are not equally dangerous:

* **False positive** (judge says correct, arithmetic says wrong) inflates the leaderboard and would
  let a genuine regression sail through the CI gate. This is the number to quote.
* **False negative** (judge says wrong, arithmetic says right) is merely pessimistic.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["AgreementReport", "cohen_kappa", "compare_verdicts"]

#: Only decidable verdicts participate. An "error" from either side means no comparison is
#: possible, so including it would conflate "the judge was wrong" with "nothing was measured".
_DECIDABLE = ("correct", "incorrect")


@dataclass(frozen=True)
class AgreementReport:
    """Judge-vs-ground-truth agreement over a set of paired verdicts."""

    n: int
    #: Pairs skipped because either side returned "error".
    n_skipped: int

    # 2x2 confusion matrix, ground truth as the rows.
    true_positive: int   # both say correct
    false_positive: int  # judge correct, truth incorrect  <- the dangerous direction
    false_negative: int  # judge incorrect, truth correct
    true_negative: int   # both say incorrect

    raw_agreement: float
    cohen_kappa: float
    false_positive_rate: float | None
    false_negative_rate: float | None
    #: Share of ground-truth "correct" labels. Near 0 or 1 means raw agreement is inflated and
    #: kappa is the only number worth quoting.
    truth_positive_rate: float | None

    @property
    def interpretation(self) -> str:
        """Conventional reading of kappa. Deliberately blunt about weak agreement."""
        k = self.cohen_kappa
        if self.n == 0:
            return "no comparable pairs"
        if k >= 0.81:
            return "almost perfect"
        if k >= 0.61:
            return "substantial"
        if k >= 0.41:
            return "moderate"
        if k >= 0.21:
            return "fair"
        if k > 0.0:
            return "slight"
        return "no better than chance"

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["interpretation"] = self.interpretation
        return out


def cohen_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    """Cohen's kappa for a 2x2 table: (p_observed - p_expected) / (1 - p_expected).

    Returns 1.0 for perfect agreement when there is no variance to explain (all pairs in one
    cell and agreeing), and 0.0 when expected agreement is already total -- the degenerate case
    where kappa is undefined and claiming credit would be wrong.
    """
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0

    p_observed = (tp + tn) / n
    # Marginals: how often each rater said "correct", independent of the other.
    judge_correct = (tp + fp) / n
    truth_correct = (tp + fn) / n
    p_expected = judge_correct * truth_correct + (1 - judge_correct) * (1 - truth_correct)

    if p_expected >= 1.0:
        return 1.0 if p_observed >= 1.0 else 0.0
    return (p_observed - p_expected) / (1 - p_expected)


def compare_verdicts(pairs: Iterable[tuple[str, str]]) -> AgreementReport:
    """Build an agreement report from (truth_verdict, judge_verdict) pairs.

    Pairs where either verdict is not decidable ("error") are counted in `n_skipped` and
    excluded -- an unextractable answer is an absence of measurement, not a judge mistake.
    """
    tp = fp = fn = tn = skipped = 0

    for truth, judge in pairs:
        if truth not in _DECIDABLE or judge not in _DECIDABLE:
            skipped += 1
            continue
        if truth == "correct":
            if judge == "correct":
                tp += 1
            else:
                fn += 1
        else:
            if judge == "correct":
                fp += 1
            else:
                tn += 1

    n = tp + fp + fn + tn
    raw = (tp + tn) / n if n else 0.0
    truth_pos = tp + fn
    truth_neg = fp + tn

    return AgreementReport(
        n=n,
        n_skipped=skipped,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        raw_agreement=raw,
        cohen_kappa=cohen_kappa(tp, fp, fn, tn),
        false_positive_rate=(fp / truth_neg) if truth_neg else None,
        false_negative_rate=(fn / truth_pos) if truth_pos else None,
        truth_positive_rate=(truth_pos / n) if n else None,
    )
