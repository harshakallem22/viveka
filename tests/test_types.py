"""Tests for core data types.

The Ollama payload below is a **real recorded response** from this machine (design §1.1),
not a hand-invented fixture, so these tests pin the actual field names and units the
provider depends on.
"""

from __future__ import annotations

import dataclasses

import pytest

from viveka.core.types import Completion, GoldenItem, Judgement, Timing

# Cold-start call to gemma3:4b, captured 2026-07-30. 93% of total was weight loading.
COLD_PAYLOAD = {
    "done": True,
    "done_reason": "stop",
    "total_duration": 5982947583,
    "load_duration": 5568053666,
    "prompt_eval_count": 18,
    "prompt_eval_duration": 338122000,
    "eval_count": 3,
    "eval_duration": 67174000,
}


class TestTimingFromOllama:
    def test_converts_nanoseconds_to_milliseconds(self):
        t = Timing.from_ollama(COLD_PAYLOAD, total_ms=5982.9, ttft_ms=5900.0)
        assert t.load_ms == pytest.approx(5568.05, abs=0.01)
        assert t.prompt_eval_ms == pytest.approx(338.12, abs=0.01)
        assert t.decode_ms == pytest.approx(67.17, abs=0.01)
        assert t.prompt_tokens == 18
        assert t.output_tokens == 3

    def test_decode_tps_excludes_model_load(self):
        """The whole point of using eval_duration: it is load-independent."""
        t = Timing.from_ollama(COLD_PAYLOAD, total_ms=5982.9, ttft_ms=5900.0)
        assert t.decode_tps == pytest.approx(44.7, abs=0.1)

    def test_wall_clock_throughput_is_wildly_misleading_on_a_cold_call(self):
        """Regression guard for the reason two throughput numbers exist (design §6.3).

        On this recorded call, reporting wall-clock tok/s as "throughput" would understate
        real decode speed by ~89x. If someone ever "simplifies" decode_tps to use total_ms,
        this test fails loudly.
        """
        t = Timing.from_ollama(COLD_PAYLOAD, total_ms=5982.9, ttft_ms=5900.0)
        assert t.end_to_end_tps == pytest.approx(0.50, abs=0.01)
        assert t.decode_tps / t.end_to_end_tps == pytest.approx(89.1, abs=1.0)

    def test_missing_counters_yield_none_not_zero_or_crash(self):
        """A failed call has no counters; None must propagate so it can be excluded from
        aggregates rather than poisoning them with zeros."""
        t = Timing.from_ollama({}, total_ms=120.0, ttft_ms=None)
        assert t.load_ms is None
        assert t.output_tokens is None
        assert t.decode_tps is None
        assert t.end_to_end_tps is None

    def test_zero_output_tokens_does_not_divide_by_zero(self):
        t = Timing(total_ms=100.0, decode_ms=0.0, output_tokens=0)
        assert t.decode_tps is None


class TestGoldenItem:
    def test_roundtrips_through_dict(self):
        item = GoldenItem(
            id="gsm8k-test-00013",
            source="gsm8k",
            category="math_reasoning",
            question="How many?",
            reference_answer="18",
            grader="numeric_exact_match",
            reference_rationale="because 9 * 2 = 18",
            metadata={"split": "test", "original_index": 13},
        )
        assert GoldenItem.from_dict(item.as_dict()) == item

    def test_omits_absent_rationale_rather_than_writing_null(self):
        item = GoldenItem(
            id="x-1", source="s", category="c", question="q",
            reference_answer="a", grader="llm_judge",
        )
        assert "reference_rationale" not in item.as_dict()
        assert GoldenItem.from_dict(item.as_dict()).reference_rationale is None

    def test_is_immutable(self):
        """Recorded eval results are historical facts; mutation would void the audit trail."""
        item = GoldenItem(
            id="x-1", source="s", category="c", question="q",
            reference_answer="a", grader="g",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            item.id = "tampered"  # type: ignore[misc]


class TestCompletionAndJudgement:
    def test_completion_ok_reflects_error(self):
        good = Completion(text="hi", timing=Timing(total_ms=1.0))
        bad = Completion(text="", timing=Timing(total_ms=1.0), error="timeout")
        assert good.ok and not bad.ok

    def test_error_verdict_is_not_scorable(self):
        """A grader that could not parse an answer must not count as a model getting it
        wrong -- otherwise a broken regex masquerades as a bad model (design §8.1)."""
        parse_failure = Judgement(
            generation_id=1, grader_name="numeric_exact_match", verdict="error", score=0.0
        )
        wrong = Judgement(
            generation_id=2, grader_name="numeric_exact_match", verdict="incorrect", score=0.0
        )
        assert not parse_failure.is_scorable
        assert wrong.is_scorable
