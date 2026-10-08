"""Tests for the SQLite store.

Focus is on the two properties the harness actually depends on: idempotent inserts
(resumability) and verbatim raw-payload retention (constraint C2). Both are the kind of thing
that silently stops working and is only noticed when a 30-minute run has to restart.
"""

from __future__ import annotations

import pytest

from viveka.core.store import Store, generation_from_row, new_run_id
from viveka.core.types import Completion, Generation, Judgement, Timing


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "test.db") as s:
        yield s


def make_gen(run_id: str, item_id="item-1", model="ollama:m", repeat=0, warmup=False):
    return Generation(
        run_id=run_id,
        item_id=item_id,
        model=model,
        repeat_idx=repeat,
        is_warmup=warmup,
        completion=Completion(
            text="the answer is 42\n#### 42",
            timing=Timing(
                total_ms=1234.5,
                ttft_ms=210.0,
                load_ms=88.0,
                prompt_eval_ms=45.0,
                decode_ms=900.0,
                prompt_tokens=101,
                output_tokens=27,
            ),
            raw={"chunks": 30, "done_reason": "stop", "final": {"eval_count": 27}},
        ),
    )


class TestRuns:
    def test_create_and_finish(self, store):
        rid = new_run_id()
        store.create_run(rid, golden_set_hash="abc123", prompt_version="v1")
        assert store.get_run(rid)["status"] == "running"
        store.finish_run(rid)
        run = store.get_run(rid)
        assert run["status"] == "complete"
        assert run["finished_utc"]
        assert run["golden_set_hash"] == "abc123"

    def test_create_run_is_idempotent_so_resume_is_safe(self, store):
        rid = new_run_id()
        store.create_run(rid, notes="first")
        store.finish_run(rid)
        store.create_run(rid, notes="second")  # resuming
        # Must not have reset the original row.
        assert store.get_run(rid)["notes"] == "first"

    def test_run_ids_are_sortable_and_unique(self):
        ids = [new_run_id() for _ in range(50)]
        assert len(set(ids)) == 50
        assert ids == sorted(ids, key=lambda s: s.split("-")[0])

    def test_latest_run_id(self, store):
        assert store.latest_run_id() is None
        first, second = new_run_id(), new_run_id()
        store.create_run(first)
        store.create_run(second)
        assert store.latest_run_id() in {first, second}


class TestGenerations:
    def test_records_all_timing_fields(self, store):
        rid = new_run_id()
        store.create_run(rid)
        gid = store.record_generation(make_gen(rid), prompt_version="v1")
        row = store.generations(rid)[0]
        assert row["id"] == gid
        assert row["ttft_ms"] == 210.0
        assert row["decode_ms"] == 900.0
        assert row["output_tokens"] == 27
        assert row["done_reason"] == "stop"
        assert row["prompt_version"] == "v1"

    def test_duplicate_insert_is_a_noop_returning_the_same_id(self, store):
        """The resumability mechanism: re-running a stage must not duplicate rows."""
        rid = new_run_id()
        store.create_run(rid)
        first = store.record_generation(make_gen(rid))
        second = store.record_generation(make_gen(rid))
        assert first == second
        assert len(store.generations(rid)) == 1

    def test_repeats_are_distinct_rows(self, store):
        rid = new_run_id()
        store.create_run(rid)
        store.record_generation(make_gen(rid, repeat=0))
        store.record_generation(make_gen(rid, repeat=1))
        assert len(store.generations(rid)) == 2

    def test_warmups_are_excluded_by_default(self, store):
        rid = new_run_id()
        store.create_run(rid)
        store.record_generation(make_gen(rid, item_id="__warmup__", warmup=True))
        store.record_generation(make_gen(rid))
        assert len(store.generations(rid)) == 1
        assert len(store.generations(rid, include_warmup=True)) == 2

    def test_completed_keys_omits_warmups(self, store):
        """A resumed run is cold again, so it must re-warm rather than skip the warmup."""
        rid = new_run_id()
        store.create_run(rid)
        store.record_generation(make_gen(rid, item_id="__warmup__", warmup=True))
        store.record_generation(make_gen(rid, item_id="item-1", repeat=0))
        assert store.completed_keys(rid) == {("item-1", 0)}

    def test_completed_keys_are_per_model(self, store):
        rid = new_run_id()
        store.create_run(rid)
        store.record_generation(make_gen(rid, model="ollama:a"))
        assert store.completed_keys(rid, model="ollama:a") == {("item-1", 0)}
        assert store.completed_keys(rid, model="ollama:b") == set()

    def test_raw_payload_survives_a_roundtrip(self, store):
        """Constraint C2: never re-run inference to recompute a metric."""
        rid = new_run_id()
        store.create_run(rid)
        store.record_generation(make_gen(rid))
        rehydrated = generation_from_row(store.generations(rid)[0])
        assert rehydrated.completion.raw["final"]["eval_count"] == 27
        assert rehydrated.completion.timing.decode_tps == pytest.approx(30.0, abs=0.1)

    def test_failed_generation_is_stored_not_dropped(self, store):
        rid = new_run_id()
        store.create_run(rid)
        store.record_generation(
            Generation(
                run_id=rid,
                item_id="item-x",
                model="ollama:m",
                repeat_idx=0,
                completion=Completion(
                    text="", timing=Timing(total_ms=120_000.0), error="ReadTimeout: too slow"
                ),
            )
        )
        row = store.generations(rid)[0]
        assert row["error"] == "ReadTimeout: too slow"
        assert not generation_from_row(row).completion.ok


class TestJudgements:
    def test_record_and_resume(self, store):
        rid = new_run_id()
        store.create_run(rid)
        gid = store.record_generation(make_gen(rid))

        j = Judgement(
            generation_id=gid,
            grader_name="numeric_exact_match",
            verdict="correct",
            score=1.0,
            reasoning="42 == 42",
        )
        first = store.record_judgement(j)
        assert store.record_judgement(j) == first  # idempotent
        assert len(store.judgements(rid)) == 1

    def test_iter_ungraded_finds_only_unjudged_generations(self, store):
        """Makes `viveka judge` resumable -- a crash mid-judging must not re-judge everything."""
        rid = new_run_id()
        store.create_run(rid)
        g1 = store.record_generation(make_gen(rid, item_id="a"))
        store.record_generation(make_gen(rid, item_id="b"))

        assert len(list(store.iter_ungraded(rid, "numeric_exact_match"))) == 2
        store.record_judgement(
            Judgement(generation_id=g1, grader_name="numeric_exact_match",
                      verdict="correct", score=1.0)
        )
        remaining = list(store.iter_ungraded(rid, "numeric_exact_match"))
        assert [r["item_id"] for r in remaining] == ["b"]

    def test_different_graders_judge_the_same_generation(self, store):
        """GSM8K items get both graders -- that pairing is what enables judge validation."""
        rid = new_run_id()
        store.create_run(rid)
        gid = store.record_generation(make_gen(rid))
        store.record_judgement(
            Judgement(generation_id=gid, grader_name="numeric_exact_match",
                      verdict="correct", score=1.0)
        )
        store.record_judgement(
            Judgement(generation_id=gid, grader_name="llm_judge", verdict="incorrect",
                      score=0.0, judge_model="qwen3:8b")
        )
        assert len(store.judgements(rid)) == 2


class TestSummaryCache:
    def test_summaries_are_replaceable(self, store):
        """run_metrics is a cache, never the source of truth -- recomputing must overwrite."""
        rid = new_run_id()
        store.create_run(rid)
        store.save_summary(rid, "ollama:m", {"accuracy": 0.5})
        store.save_summary(rid, "ollama:m", {"accuracy": 0.6})
        assert store.summaries(rid) == [{"accuracy": 0.6}]
