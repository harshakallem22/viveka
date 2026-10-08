"""Aggregation: raw generations -> reportable metrics.

Stdlib only (design §4.1). This module is pure arithmetic over recorded results, which is why
it can be re-run for free whenever the definitions change (constraint C2).

Three deliberate choices worth knowing before reading:

* **Percentiles are nearest-rank, never interpolated** (§6.4). Interpolation invents a latency
  that was never observed; SLO convention is to report a real sample.
* **Accuracy always carries a confidence interval** (§6.5). At n=50 a bare percentage overstates
  precision badly enough to make two models look different when they are not.
* **Errors and unscorable items are counted, not silently dropped.** A model that times out on
  half the set must not look fast, and a grader that fails to parse must not look like a model
  that got the answer wrong.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from viveka.core.types import Generation, GoldenItem, Judgement

__all__ = [
    "ModelSummary",
    "percentile",
    "summarize",
    "wilson_interval",
]


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile: returns an actually-observed value.

    Definition: on values sorted ascending, index = ceil(p/100 * n) - 1, clamped.

    Be aware how coarse this is at small n -- at n=50, P95 resolves to the 48th of 50 samples,
    i.e. the third largest. Callers should report `n` alongside any percentile so the reader can
    judge how much to trust it (§6.4).
    """
    if not values:
        return None
    if not 0 < p <= 100:
        raise ValueError(f"percentile p must be in (0, 100], got {p}")
    ordered = sorted(values)
    idx = math.ceil(p / 100 * len(ordered)) - 1
    return ordered[max(0, min(idx, len(ordered) - 1))]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%).

    Preferred over the normal approximation because it stays inside [0, 1] and behaves sensibly
    at small n and near 0 or 1 -- exactly our regime. A worked check: 30/50 gives roughly
    [0.46, 0.72], which is the honest way to say "60%" on 50 items.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


@dataclass(frozen=True)
class ModelSummary:
    """Everything reportable about one model in one run."""

    model: str
    n_items: int
    n_generations: int
    n_errors: int
    n_truncated: int

    # Correctness. None until judgements exist (Milestone 3).
    n_scored: int = 0
    n_correct: int = 0
    n_unscorable: int = 0
    accuracy: float | None = None
    accuracy_ci_low: float | None = None
    accuracy_ci_high: float | None = None

    # Latency, in milliseconds. Warmups and errored calls excluded.
    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_min_ms: float | None = None
    latency_max_ms: float | None = None

    # Throughput, tokens/sec.
    decode_tps_p50: float | None = None
    decode_tps_mean: float | None = None
    end_to_end_tps_p50: float | None = None

    #: Model load time from the warmup call: the cost that warmup exists to keep out of the
    #: percentiles above.
    cold_start_ms: float | None = None

    #: Accuracy split by item category, e.g. {"math_reasoning": 0.92, "factual_qa": 0.7}.
    accuracy_by_category: dict[str, float] = field(default_factory=dict)

    @property
    def accuracy_ci_width(self) -> float | None:
        """How much precision we actually have. Two models closer than this are not separable."""
        if self.accuracy_ci_low is None or self.accuracy_ci_high is None:
            return None
        return self.accuracy_ci_high - self.accuracy_ci_low

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize(
    model: str,
    generations: Iterable[Generation],
    *,
    judgements: Iterable[Judgement] | None = None,
    items: Mapping[str, GoldenItem] | None = None,
    grader_name: str | None = None,
) -> ModelSummary:
    """Aggregate one model's generations (and optionally its judgements) into a summary.

    Args:
        model: provider name, e.g. "ollama:llama3.1:8b".
        generations: this model's generations. Warmups may be included -- they are used only for
            `cold_start_ms` and are excluded from every distribution.
        judgements: scores for those generations, joined on `generation_id`. Omit before
            Milestone 3; correctness fields then stay None rather than defaulting to zero,
            because "unmeasured" and "zero" must not look alike.
        items: golden items by id, needed for the per-category accuracy breakdown.
        grader_name: restrict scoring to one grader. Required when both a deterministic grader
            and the LLM judge have scored the same generations, or accuracy would double-count.
    """
    gens = list(generations)
    warmups = [g for g in gens if g.is_warmup]
    scored_gens = [g for g in gens if not g.is_warmup]

    ok = [g for g in scored_gens if g.completion.ok]
    n_errors = len(scored_gens) - len(ok)
    n_truncated = sum(1 for g in ok if g.completion.raw.get("done_reason") == "length")

    ttfts = [g.completion.timing.ttft_ms for g in ok if g.completion.timing.ttft_ms is not None]
    latencies = [g.completion.timing.total_ms for g in ok if g.completion.timing.total_ms]
    decode = [
        tps for g in ok if (tps := g.completion.timing.decode_tps) is not None
    ]
    e2e = [tps for g in ok if (tps := g.completion.timing.end_to_end_tps) is not None]

    cold_start = next(
        (g.completion.timing.load_ms for g in warmups if g.completion.timing.load_ms), None
    )

    summary_kwargs: dict[str, Any] = {
        "model": model,
        "n_items": len({g.item_id for g in scored_gens}),
        "n_generations": len(scored_gens),
        "n_errors": n_errors,
        "n_truncated": n_truncated,
        "ttft_p50_ms": percentile(ttfts, 50),
        "ttft_p95_ms": percentile(ttfts, 95),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_min_ms": min(latencies) if latencies else None,
        "latency_max_ms": max(latencies) if latencies else None,
        "decode_tps_p50": percentile(decode, 50),
        "decode_tps_mean": _mean(decode),
        "end_to_end_tps_p50": percentile(e2e, 50),
        "cold_start_ms": cold_start,
    }

    if judgements is None:
        return ModelSummary(**summary_kwargs)

    by_gen_id = {g.id: g for g in scored_gens if g.id is not None}
    relevant = [
        j
        for j in judgements
        if j.generation_id in by_gen_id
        and (grader_name is None or j.grader_name == grader_name)
    ]

    # A grader that could not parse an answer is a harness failure, not a wrong model. Counting
    # it as incorrect would make a broken extraction rule look like poor model quality (§8.1).
    scorable = [j for j in relevant if j.is_scorable]
    n_correct = sum(1 for j in scorable if j.verdict == "correct")
    n_scored = len(scorable)

    accuracy: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    if n_scored:
        accuracy = n_correct / n_scored
        ci_low, ci_high = wilson_interval(n_correct, n_scored)

    by_category: dict[str, float] = {}
    if items:
        buckets: dict[str, list[bool]] = {}
        for j in scorable:
            gen = by_gen_id[j.generation_id]
            item = items.get(gen.item_id)
            if item is None:
                continue
            buckets.setdefault(item.category, []).append(j.verdict == "correct")
        by_category = {
            cat: sum(flags) / len(flags) for cat, flags in sorted(buckets.items()) if flags
        }

    summary_kwargs.update(
        n_scored=n_scored,
        n_correct=n_correct,
        n_unscorable=len(relevant) - n_scored,
        accuracy=accuracy,
        accuracy_ci_low=ci_low,
        accuracy_ci_high=ci_high,
        accuracy_by_category=by_category,
    )
    return ModelSummary(**summary_kwargs)
