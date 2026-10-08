"""Tests for CI regression gating.

The gate's whole value is that it fails when it should. A gate nobody has watched fail is not
evidence of anything, so most of these assert on *blocking* outcomes.
"""

from __future__ import annotations

from viveka.core.gate import (
    GateThresholds,
    evaluate_gate,
    summarize_markdown,
    thresholds_from_mapping,
)


def results(
    *,
    accuracy: float = 0.9,
    p95: float = 20000.0,
    verdicts: dict[str, str] | None = None,
    golden_hash: str = "abc123",
    prompt_version: str = "v1",
    model: str = "ollama:m",
) -> dict:
    payload = {
        "run_id": "r1",
        "golden_set_hash": golden_hash,
        "prompt_version": prompt_version,
        "models": [
            {
                "model": model,
                "accuracy": accuracy,
                "latency_p95_ms": p95,
                "decode_tps_p50": 20.0,
                "n_generations": 50,
            }
        ],
    }
    if verdicts is not None:
        payload["item_verdicts"] = {model: verdicts}
    return payload


LOOSE = GateThresholds(min_accuracy=0.4, max_p95_latency_ms=45000, max_accuracy_drop=0.1,
                       max_regressed_items=3)


class TestAbsoluteFloors:
    def test_passes_when_above_floors(self):
        assert evaluate_gate(results(), None, LOOSE).passed

    def test_fails_below_min_accuracy(self):
        report = evaluate_gate(results(accuracy=0.30), None, LOOSE)
        assert not report.passed
        assert any(c.name == "min_accuracy" for c in report.failures)

    def test_fails_above_p95_ceiling(self):
        report = evaluate_gate(results(p95=60000.0), None, LOOSE)
        assert not report.passed
        assert any(c.name == "max_p95_latency" for c in report.failures)

    def test_missing_accuracy_warns_rather_than_blocks(self):
        """Before `viveka judge` has run there is nothing to gate on; that should not read as a
        regression."""
        payload = results()
        payload["models"][0]["accuracy"] = None
        report = evaluate_gate(payload, None, LOOSE)
        assert report.passed
        assert any(c.name == "accuracy_measured" for c in report.warnings)

    def test_empty_results_fail(self):
        assert not evaluate_gate({"models": []}, None, LOOSE).passed


class TestBaselineComparison:
    def test_accuracy_drop_within_tolerance_passes(self):
        report = evaluate_gate(results(accuracy=0.85), results(accuracy=0.90), LOOSE)
        assert report.passed

    def test_accuracy_drop_beyond_tolerance_fails(self):
        report = evaluate_gate(results(accuracy=0.75), results(accuracy=0.90), LOOSE)
        assert not report.passed
        failure = next(c for c in report.failures if c.name == "max_accuracy_drop")
        assert "15.0%" in failure.detail

    def test_improvement_never_fails(self):
        report = evaluate_gate(results(accuracy=0.99), results(accuracy=0.90), LOOSE)
        assert report.passed
        check = next(c for c in report.checks if c.name == "max_accuracy_drop")
        assert "gain" in check.detail


class TestAntiTamper:
    def test_changed_eval_set_blocks_the_comparison(self):
        """Deleting the three hardest items to turn a red build green must not work: a different
        eval set makes accuracy deltas meaningless, not merely suspicious."""
        report = evaluate_gate(
            results(golden_hash="tampered"), results(golden_hash="abc123"), LOOSE
        )
        assert not report.passed
        assert any(c.name == "golden_set_unchanged" for c in report.failures)

    def test_eval_set_check_can_be_disabled(self):
        thresholds = GateThresholds(min_accuracy=0.4, require_same_golden_set=False)
        report = evaluate_gate(
            results(golden_hash="different"), results(golden_hash="abc123"), thresholds
        )
        assert report.passed

    def test_prompt_version_change_warns(self):
        """A prompt edit can explain an accuracy change entirely, so it must be surfaced -- but it
        is a legitimate thing to do, so it warns rather than blocks."""
        report = evaluate_gate(results(prompt_version="v2"), results(prompt_version="v1"), LOOSE)
        assert report.passed
        assert any(c.name == "prompt_version_unchanged" for c in report.warnings)

    def test_dropping_a_model_fails(self):
        """Removing a failing model is not the same as fixing it."""
        candidate = results(model="ollama:a")
        baseline = {
            "golden_set_hash": "abc123",
            "prompt_version": "v1",
            "models": [
                {"model": "ollama:a", "accuracy": 0.9, "latency_p95_ms": 1000.0},
                {"model": "ollama:b", "accuracy": 0.9, "latency_p95_ms": 1000.0},
            ],
        }
        report = evaluate_gate(candidate, baseline, LOOSE)
        assert not report.passed
        failure = next(c for c in report.failures if c.name == "no_models_dropped")
        assert "ollama:b" in failure.detail


class TestItemLevelRegression:
    """Item-level gating is more sensitive than an aggregate delta, and it names what broke."""

    def test_counts_and_names_regressed_items(self):
        baseline = results(verdicts={f"i{n}": "correct" for n in range(10)})
        flipped = {f"i{n}": ("incorrect" if n < 4 else "correct") for n in range(10)}
        candidate = results(accuracy=0.6, verdicts=flipped)

        report = evaluate_gate(candidate, baseline, LOOSE)
        assert not report.passed
        failure = next(c for c in report.failures if c.name == "max_regressed_items")
        assert "4 item(s)" in failure.detail
        assert report.regressions["ollama:m"] == ["i0", "i1", "i2", "i3"]

    def test_within_limit_passes(self):
        baseline = results(verdicts={f"i{n}": "correct" for n in range(10)})
        candidate = results(
            accuracy=0.8, verdicts={f"i{n}": ("incorrect" if n < 2 else "correct") for n in range(10)}
        )
        report = evaluate_gate(candidate, baseline, LOOSE)
        assert report.passed
        assert report.regressions["ollama:m"] == ["i0", "i1"]

    def test_improvements_are_tracked_separately(self):
        baseline = results(verdicts={"i0": "incorrect", "i1": "incorrect"})
        candidate = results(verdicts={"i0": "correct", "i1": "correct"})
        report = evaluate_gate(candidate, baseline, LOOSE)
        assert report.passed
        assert report.improvements["ollama:m"] == ["i0", "i1"]
        assert "ollama:m" not in report.regressions

    def test_offsetting_changes_are_caught_despite_flat_accuracy(self):
        """The reason item-level gating exists: 4 items break, 4 improve, aggregate accuracy is
        unchanged, and an aggregate-only gate sees nothing wrong."""
        baseline = results(
            accuracy=0.5,
            verdicts={**{f"a{n}": "correct" for n in range(4)}, **{f"b{n}": "incorrect" for n in range(4)}},
        )
        candidate = results(
            accuracy=0.5,
            verdicts={**{f"a{n}": "incorrect" for n in range(4)}, **{f"b{n}": "correct" for n in range(4)}},
        )
        report = evaluate_gate(candidate, baseline, LOOSE)
        assert not report.passed
        assert len(report.regressions["ollama:m"]) == 4
        drop_check = next(c for c in report.checks if c.name == "max_accuracy_drop")
        assert drop_check.passed  # aggregate looks fine...
        item_check = next(c for c in report.checks if c.name == "max_regressed_items")
        assert not item_check.passed  # ...but the items say otherwise

    def test_error_verdicts_do_not_count_as_regressions(self):
        baseline = results(verdicts={"i0": "correct"})
        candidate = results(verdicts={"i0": "error"})
        report = evaluate_gate(candidate, baseline, LOOSE)
        assert report.regressions == {}


class TestThresholdsFromMapping:
    def test_reads_known_keys_and_ignores_the_rest(self):
        t = thresholds_from_mapping({"min_accuracy": 0.4, "unrelated": 1, "max_regressed_items": 7})
        assert t.min_accuracy == 0.4
        assert t.max_regressed_items == 7
        assert t.max_p95_latency_ms == GateThresholds().max_p95_latency_ms

    def test_none_yields_defaults(self):
        assert thresholds_from_mapping(None) == GateThresholds()


class TestMarkdownSummary:
    def test_reports_verdict_and_deltas(self):
        candidate = results(accuracy=0.75, p95=30000.0)
        baseline = results(accuracy=0.90, p95=20000.0)
        report = evaluate_gate(candidate, baseline, LOOSE)
        md = summarize_markdown(candidate, baseline, report)
        assert "**FAILED**" in md
        assert "-15.0pp" in md
        assert "+10.00s" in md
        assert "Blocking failures" in md

    def test_passing_summary(self):
        md = summarize_markdown(results(), None, evaluate_gate(results(), None, LOOSE))
        assert "**PASSED**" in md
        assert "Blocking failures" not in md
