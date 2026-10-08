"""Tests for run orchestration, using a fake Provider.

No Ollama needed: the Provider protocol is the seam, so a fake implementation exercises the
entire orchestration path. That this is easy is itself evidence the seam is in the right place
(design §5.2) -- and it is what will let the CI pipeline run the full stack in seconds.
"""

from __future__ import annotations

import pytest

from viveka.core.prompts import PROMPT_VERSION
from viveka.core.runner import RunConfig, Runner, host_info
from viveka.core.store import Store
from viveka.core.types import Completion, GoldenItem, Timing


class FakeProvider:
    """Records the order of calls so we can assert on batching and lifecycle."""

    def __init__(self, model: str, *, fail_on: set[str] | None = None, log: list | None = None):
        self.model = model
        self.fail_on = fail_on or set()
        self.log: list[str] = log if log is not None else []
        self.prompts: list[str] = []
        self.warmups = 0
        self.unloads = 0

    @property
    def name(self) -> str:
        return f"fake:{self.model}"

    def complete(self, prompt: str, *, system=None, **overrides) -> Completion:
        self.prompts.append(prompt)
        self.log.append(f"complete:{self.model}")
        if any(token in prompt for token in self.fail_on):
            return Completion(
                text="", timing=Timing(total_ms=5.0), error="RuntimeError: injected failure"
            )
        return Completion(
            text="answer\n#### 7",
            timing=Timing(
                total_ms=100.0, ttft_ms=10.0, decode_ms=80.0, output_tokens=8, prompt_tokens=20
            ),
            raw={"done_reason": "stop"},
        )

    def warmup(self) -> Completion:
        self.warmups += 1
        self.log.append(f"warmup:{self.model}")
        return Completion(
            text="", timing=Timing(total_ms=4300.0, load_ms=4270.0), raw={"done_reason": "stop"}
        )

    def unload(self) -> None:
        self.unloads += 1
        self.log.append(f"unload:{self.model}")


def items(n: int = 3) -> list[GoldenItem]:
    return [
        GoldenItem(
            id=f"gsm8k-test-{i:05d}",
            source="gsm8k",
            category="math_reasoning",
            question=f"question {i}?",
            reference_answer="7",
            grader="numeric_exact_match",
        )
        for i in range(n)
    ]


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "runner.db") as s:
        yield s


class TestBatching:
    def test_models_run_to_completion_one_at_a_time(self, store):
        """The core scheduling requirement (design §1.4).

        Interleaving models would make Ollama evict and reload 5GB of weights on nearly every
        call. Asserting on call order pins the behaviour that makes runs finish in minutes
        rather than hours.
        """
        log: list[str] = []
        a, b = FakeProvider("a", log=log), FakeProvider("b", log=log)
        Runner(store, config=RunConfig(warmup=True)).run(items(3), [a, b])

        assert log == [
            "warmup:a", "complete:a", "complete:a", "complete:a", "unload:a",
            "warmup:b", "complete:b", "complete:b", "complete:b", "unload:b",
        ]

    def test_each_model_is_unloaded_after_its_batch(self, store):
        a, b = FakeProvider("a"), FakeProvider("b")
        Runner(store).run(items(2), [a, b])
        assert a.unloads == 1 and b.unloads == 1


class TestWarmup:
    def test_warmup_is_recorded_but_excluded_from_results(self, store):
        """Cold-start cost stays visible in the data while staying out of the percentiles."""
        p = FakeProvider("a")
        rid = Runner(store, config=RunConfig(warmup=True)).run(items(3), [p])

        assert len(store.generations(rid)) == 3
        all_rows = store.generations(rid, include_warmup=True)
        assert len(all_rows) == 4
        warm = [r for r in all_rows if r["is_warmup"]]
        assert len(warm) == 1 and warm[0]["item_id"] == "__warmup__"

    def test_cold_start_cost_is_persisted(self, store):
        """"First-token cost after eviction" is a publishable number, so warmup() returning a
        Completion (rather than None) must actually reach the database."""
        rid = Runner(store, config=RunConfig(warmup=True)).run(items(1), [FakeProvider("a")])
        warm = [r for r in store.generations(rid, include_warmup=True) if r["is_warmup"]]
        assert warm[0]["load_ms"] == 4270.0

    def test_warmup_can_be_disabled(self, store):
        p = FakeProvider("a")
        rid = Runner(store, config=RunConfig(warmup=False)).run(items(2), [p])
        assert len(store.generations(rid, include_warmup=True)) == 2


class TestRepeats:
    def test_repeats_multiply_generations(self, store):
        p = FakeProvider("a")
        rid = Runner(store, config=RunConfig(repeats=3, warmup=False)).run(items(4), [p])
        rows = store.generations(rid)
        assert len(rows) == 12
        assert {r["repeat_idx"] for r in rows} == {0, 1, 2}


class TestErrorIsolation:
    def test_one_failure_does_not_abort_the_run(self, store):
        """A timeout on item 2 must not discard the results already paid for."""
        p = FakeProvider("a", fail_on={"question 1?"})
        rid = Runner(store, config=RunConfig(warmup=False)).run(items(3), [p])

        rows = store.generations(rid)
        assert len(rows) == 3
        failed = [r for r in rows if r["error"]]
        assert len(failed) == 1
        assert "injected failure" in failed[0]["error"]
        assert store.get_run(rid)["status"] == "complete"


class TestResume:
    def test_resuming_skips_completed_work(self, store):
        first = FakeProvider("a")
        rid = Runner(store, config=RunConfig(warmup=False)).run(items(3), [first])
        assert len(first.prompts) == 3

        second = FakeProvider("a")
        Runner(store, config=RunConfig(warmup=False, resume=True)).run(
            items(3), [second], run_id=rid
        )
        assert second.prompts == []  # nothing re-generated
        assert len(store.generations(rid)) == 3

    def test_resume_extends_a_partial_run(self, store):
        p1 = FakeProvider("a")
        rid = Runner(store, config=RunConfig(warmup=False)).run(items(2), [p1])

        p2 = FakeProvider("a")
        Runner(store, config=RunConfig(warmup=False)).run(items(4), [p2], run_id=rid)
        assert len(p2.prompts) == 2  # only the two new items
        assert len(store.generations(rid)) == 4

    def test_resume_can_be_turned_off(self, store):
        p1 = FakeProvider("a")
        rid = Runner(store, config=RunConfig(warmup=False)).run(items(2), [p1])
        p2 = FakeProvider("a")
        Runner(store, config=RunConfig(warmup=False, resume=False)).run(
            items(2), [p2], run_id=rid
        )
        assert len(p2.prompts) == 2  # re-generated, but INSERT OR IGNORE keeps rows at 2
        assert len(store.generations(rid)) == 2


class TestProvenance:
    def test_run_records_what_produced_it(self, store):
        p = FakeProvider("a")
        rid = Runner(store, config=RunConfig(warmup=False)).run(
            items(1), [p], golden_set_hash="deadbeef", ollama_version="0.32.5"
        )
        run = store.get_run(rid)
        assert run["golden_set_hash"] == "deadbeef"
        assert run["ollama_version"] == "0.32.5"
        assert run["prompt_version"] == PROMPT_VERSION
        assert run["viveka_version"]
        assert run["host_json"]

    def test_prompt_version_is_recorded_per_generation(self, store):
        """The CI gate exists to catch prompt changes, so an unversioned result is unusable."""
        rid = Runner(store, config=RunConfig(warmup=False)).run(items(1), [FakeProvider("a")])
        assert store.generations(rid)[0]["prompt_version"] == PROMPT_VERSION

    def test_host_info_is_populated(self):
        info = host_info()
        assert info["platform"] and info["machine"] and info["python"]


class TestProgressEvents:
    def test_events_cover_warmup_items_and_skips(self, store):
        events = []
        rid = Runner(
            store, config=RunConfig(warmup=True), on_progress=events.append
        ).run(items(2), [FakeProvider("a")])

        assert sum(1 for e in events if e.is_warmup) == 1
        assert sum(1 for e in events if not e.is_warmup) == 2
        assert all(e.item_total == 2 for e in events if not e.is_warmup)

        events2 = []
        Runner(store, config=RunConfig(warmup=False), on_progress=events2.append).run(
            items(2), [FakeProvider("a")], run_id=rid
        )
        assert all(e.skipped for e in events2)

    def test_failed_flag_reflects_completion_error(self, store):
        events = []
        Runner(store, config=RunConfig(warmup=False), on_progress=events.append).run(
            items(2), [FakeProvider("a", fail_on={"question 0?"})]
        )
        assert [e.failed for e in events] == [True, False]


class TestPromptRendering:
    def test_math_items_get_the_answer_marker_instruction(self, store):
        """Extraction reliability depends on this instruction reaching the model."""
        p = FakeProvider("a")
        Runner(store, config=RunConfig(warmup=False)).run(items(1), [p])
        assert "#### <number>" in p.prompts[0]
        assert "question 0?" in p.prompts[0]
