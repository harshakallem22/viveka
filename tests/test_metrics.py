"""Tests for the metrics layer.

Expected values are hand-computed or derived from the design doc, not captured from the
implementation -- a test that merely records current behaviour cannot catch a wrong formula.
"""

from __future__ import annotations

import math

import pytest

from viveka.core.metrics import ModelSummary, percentile, summarize, wilson_interval
from viveka.core.types import Completion, Generation, GoldenItem, Judgement, Timing


class TestPercentile:
    def test_returns_an_observed_value_not_an_interpolation(self):
        """Nearest-rank by design: an interpolated latency was never actually measured."""
        values = [1.0, 2.0, 3.0, 4.0]
        assert percentile(values, 50) in values
        # Linear interpolation would give 2.5 for the median here.
        assert percentile(values, 50) == 2.0

    @pytest.mark.parametrize(
        ("p", "expected"),
        [
            (50, 5.0),   # ceil(0.50*10)=5  -> index 4 -> 5th smallest
            (90, 9.0),   # ceil(0.90*10)=9  -> index 8
            (95, 10.0),  # ceil(0.95*10)=10 -> index 9 -> the max, at n=10
            (100, 10.0),
        ],
    )
    def test_known_ranks_on_one_to_ten(self, p, expected):
        assert percentile([float(i) for i in range(1, 11)], p) == expected

    def test_p95_at_n_50_is_the_third_largest(self):
        """Documents how coarse P95 is at our sample size (design §6.4).

        ceil(0.95*50) = 48, so P95 is the 48th of 50 -- third from the top. Worth pinning: if
        someone swaps in an interpolating percentile, the reported tail silently changes.
        """
        values = [float(i) for i in range(1, 51)]
        assert percentile(values, 95) == 48.0

    def test_ignores_input_order(self):
        assert percentile([9.0, 1.0, 5.0], 50) == percentile([1.0, 5.0, 9.0], 50)

    def test_single_value(self):
        assert percentile([42.0], 95) == 42.0

    def test_empty_returns_none_rather_than_raising(self):
        """A model whose every call failed has no latencies; that must not crash a report."""
        assert percentile([], 50) is None

    @pytest.mark.parametrize("bad", [0, -5, 101])
    def test_rejects_out_of_range_p(self, bad):
        with pytest.raises(ValueError, match="percentile"):
            percentile([1.0, 2.0], bad)


class TestWilsonInterval:
    def test_matches_the_worked_example_in_the_design_doc(self):
        """p=0.60 at n=30 -> roughly [0.42, 0.75] (design §6.5)."""
        low, high = wilson_interval(18, 30)
        assert low == pytest.approx(0.423, abs=0.001)
        assert high == pytest.approx(0.754, abs=0.001)

    def test_interval_narrows_as_n_grows(self):
        """The reason the golden set is 60 items rather than 35."""
        narrow = wilson_interval(30, 50)
        wide = wilson_interval(18, 30)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_stays_within_zero_and_one_at_the_extremes(self):
        """Where the normal approximation would escape [0,1] and report nonsense."""
        assert wilson_interval(0, 10)[0] == 0.0
        assert wilson_interval(10, 10)[1] == 1.0
        assert 0.0 <= wilson_interval(0, 10)[1] <= 1.0
        assert 0.0 <= wilson_interval(10, 10)[0] <= 1.0

    def test_perfect_score_still_has_a_lower_bound_below_one(self):
        """50/50 is not proof of 100% -- the interval must say so."""
        low, high = wilson_interval(50, 50)
        assert high == 1.0
        assert low < 1.0

    def test_zero_n_does_not_divide_by_zero(self):
        assert wilson_interval(0, 0) == (0.0, 0.0)

    def test_contains_the_point_estimate(self):
        for k, n in [(1, 10), (5, 10), (9, 10), (23, 50), (46, 50)]:
            low, high = wilson_interval(k, n)
            assert low <= k / n <= high


# --- helpers ------------------------------------------------------------------------------


def gen(
    item_id: str,
    *,
    gid: int | None = None,
    total_ms: float = 1000.0,
    ttft_ms: float | None = 100.0,
    decode_ms: float | None = 800.0,
    output_tokens: int | None = 16,
    load_ms: float | None = None,
    error: str | None = None,
    done_reason: str = "stop",
    warmup: bool = False,
) -> Generation:
    return Generation(
        run_id="r1",
        item_id=item_id,
        model="ollama:m",
        repeat_idx=0,
        is_warmup=warmup,
        id=gid,
        completion=Completion(
            text="#### 7",
            timing=Timing(
                total_ms=total_ms,
                ttft_ms=ttft_ms,
                decode_ms=decode_ms,
                output_tokens=output_tokens,
                load_ms=load_ms,
            ),
            raw={"done_reason": done_reason},
            error=error,
        ),
    )


def item(item_id: str, category: str = "math_reasoning") -> GoldenItem:
    return GoldenItem(
        id=item_id,
        source="gsm8k",
        category=category,
        question="q",
        reference_answer="7",
        grader="numeric_exact_match",
    )


def judged(gid: int, verdict: str, grader: str = "numeric_exact_match") -> Judgement:
    return Judgement(
        generation_id=gid,
        grader_name=grader,
        verdict=verdict,  # type: ignore[arg-type]
        score=1.0 if verdict == "correct" else 0.0,
    )


# --- summarize ----------------------------------------------------------------------------


class TestSummarizeLatency:
    def test_warmups_are_excluded_from_every_distribution(self):
        """The single most important exclusion: a cold call is ~6.8x slower and would dominate
        P95 at these sample sizes (design §6.1)."""
        gens = [
            gen("__warmup__", total_ms=6000.0, load_ms=5570.0, warmup=True),
            gen("a", total_ms=1000.0),
            gen("b", total_ms=1100.0),
        ]
        s = summarize("ollama:m", gens)
        assert s.n_generations == 2
        assert s.latency_max_ms == 1100.0  # not 6000
        assert s.cold_start_ms == 5570.0  # but the cold cost is still reported

    def test_errored_calls_excluded_from_latency_but_counted(self):
        """A model that times out on half the set must not thereby look fast."""
        gens = [
            gen("a", total_ms=1000.0),
            gen("b", total_ms=120_000.0, error="ReadTimeout", ttft_ms=None, decode_ms=None),
        ]
        s = summarize("ollama:m", gens)
        assert s.n_errors == 1
        assert s.n_generations == 2
        assert s.latency_max_ms == 1000.0

    def test_percentiles_and_extremes(self):
        gens = [gen(f"i{i}", total_ms=float(i) * 100) for i in range(1, 11)]
        s = summarize("ollama:m", gens)
        assert s.latency_p50_ms == 500.0
        assert s.latency_p95_ms == 1000.0
        assert s.latency_min_ms == 100.0
        assert s.latency_max_ms == 1000.0

    def test_truncated_responses_are_counted(self):
        gens = [gen("a"), gen("b", done_reason="length"), gen("c", done_reason="length")]
        assert summarize("ollama:m", gens).n_truncated == 2

    def test_throughput_uses_decode_time_not_wall_clock(self):
        """16 tokens in 800ms decode = 20 tok/s, regardless of the 4s wall clock (design §6.3)."""
        s = summarize("ollama:m", [gen("a", total_ms=4000.0, decode_ms=800.0, output_tokens=16)])
        assert s.decode_tps_p50 == pytest.approx(20.0)
        assert s.end_to_end_tps_p50 == pytest.approx(4.0)

    def test_n_items_counts_distinct_items_not_generations(self):
        """With repeats, n_generations and n_items diverge."""
        gens = [gen("a"), gen("a"), gen("b")]
        s = summarize("ollama:m", gens)
        assert s.n_generations == 3
        assert s.n_items == 2

    def test_all_failed_yields_none_metrics_not_zeros(self):
        """None means "not measured"; 0.0 would mean "instant", which is a lie."""
        s = summarize("ollama:m", [gen("a", error="boom", ttft_ms=None, decode_ms=None)])
        assert s.latency_p50_ms is None
        assert s.decode_tps_p50 is None
        assert s.n_errors == 1

    def test_empty_input(self):
        s = summarize("ollama:m", [])
        assert s.n_generations == 0 and s.latency_p50_ms is None


class TestSummarizeAccuracy:
    def test_accuracy_is_none_before_any_judgements(self):
        """Milestone 2 reports latency only. Unmeasured must not look like zero."""
        s = summarize("ollama:m", [gen("a", gid=1)])
        assert s.accuracy is None
        assert s.n_scored == 0

    def test_accuracy_with_confidence_interval(self):
        gens = [gen(f"i{i}", gid=i) for i in range(1, 11)]
        judgements = [judged(i, "correct" if i <= 6 else "incorrect") for i in range(1, 11)]
        s = summarize("ollama:m", gens, judgements=judgements)
        assert s.accuracy == pytest.approx(0.6)
        assert s.n_correct == 6 and s.n_scored == 10
        assert s.accuracy_ci_low is not None and s.accuracy_ci_low < 0.6 < s.accuracy_ci_high

    def test_grader_errors_are_excluded_from_the_denominator(self):
        """The heart of the error-vs-incorrect split: a grader that could not parse an answer
        must not drag accuracy down as though the model was wrong (design §8.1)."""
        gens = [gen(f"i{i}", gid=i) for i in range(1, 5)]
        judgements = [
            judged(1, "correct"),
            judged(2, "correct"),
            judged(3, "incorrect"),
            judged(4, "error"),  # extraction failure, not a model failure
        ]
        s = summarize("ollama:m", gens, judgements=judgements)
        assert s.n_scored == 3
        assert s.n_unscorable == 1
        assert s.accuracy == pytest.approx(2 / 3)  # not 2/4

    def test_grader_filter_prevents_double_counting(self):
        """GSM8K items get scored by both graders; without filtering, accuracy is meaningless."""
        gens = [gen("a", gid=1)]
        judgements = [
            judged(1, "correct", grader="numeric_exact_match"),
            judged(1, "incorrect", grader="llm_judge"),
        ]
        assert summarize("ollama:m", gens, judgements=judgements).n_scored == 2
        strict = summarize(
            "ollama:m", gens, judgements=judgements, grader_name="numeric_exact_match"
        )
        assert strict.n_scored == 1 and strict.accuracy == 1.0

    def test_judgements_for_other_models_are_ignored(self):
        gens = [gen("a", gid=1)]
        judgements = [judged(1, "correct"), judged(999, "incorrect")]
        assert summarize("ollama:m", gens, judgements=judgements).n_scored == 1

    def test_accuracy_by_category(self):
        gens = [gen("m1", gid=1), gen("m2", gid=2), gen("f1", gid=3), gen("f2", gid=4)]
        items = {
            "m1": item("m1", "math_reasoning"),
            "m2": item("m2", "math_reasoning"),
            "f1": item("f1", "factual_qa"),
            "f2": item("f2", "factual_qa"),
        }
        judgements = [
            judged(1, "correct"),
            judged(2, "correct"),
            judged(3, "correct"),
            judged(4, "incorrect"),
        ]
        s = summarize("ollama:m", gens, judgements=judgements, items=items)
        assert s.accuracy_by_category == {"factual_qa": 0.5, "math_reasoning": 1.0}

    def test_warmup_generations_never_get_scored(self):
        gens = [gen("__warmup__", gid=1, warmup=True), gen("a", gid=2)]
        judgements = [judged(1, "incorrect"), judged(2, "correct")]
        s = summarize("ollama:m", gens, judgements=judgements)
        assert s.n_scored == 1 and s.accuracy == 1.0


class TestConfidenceIntervalWidth:
    def test_width_exposes_whether_two_models_are_separable(self):
        gens = [gen(f"i{i}", gid=i) for i in range(1, 51)]
        judgements = [judged(i, "correct" if i <= 30 else "incorrect") for i in range(1, 51)]
        s = summarize("ollama:m", gens, judgements=judgements)
        assert s.accuracy_ci_width is not None
        # 30/50 -> roughly [0.46, 0.72]: a ~26-point spread at our real sample size.
        assert 0.24 < s.accuracy_ci_width < 0.28

    def test_width_is_none_when_unmeasured(self):
        assert summarize("ollama:m", [gen("a")]).accuracy_ci_width is None


class TestSerialization:
    def test_as_dict_roundtrips_through_the_dataclass(self):
        s = summarize("ollama:m", [gen("a", total_ms=1000.0)])
        d = s.as_dict()
        assert d["model"] == "ollama:m"
        assert ModelSummary(**d) == s

    def test_as_dict_is_json_serializable(self):
        import json

        gens = [gen("a", gid=1)]
        s = summarize("ollama:m", gens, judgements=[judged(1, "correct")], items={"a": item("a")})
        assert json.loads(json.dumps(s.as_dict()))["accuracy"] == 1.0


class TestAgainstRealBenchmarkNumbers:
    """Reproduces the published Milestone 1 figures for gemma3:4b, so a change in the
    percentile definition would visibly break the documented result."""

    def test_p95_rank_at_n_60(self):
        # ceil(0.95*60) = 57 -> the 57th of 60, i.e. 4th largest.
        values = [float(i) for i in range(1, 61)]
        assert percentile(values, 95) == 57.0
        assert math.ceil(0.95 * 60) == 57
