"""Integrity tests for the committed golden set.

These run in CI on every PR and cost nothing (no models, no network). They exist because the
CI regression gate is only meaningful if the eval set provably did not change between the
baseline run and the candidate run (design §7.2, §11.2).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from viveka.core.datasets import (
    GoldenSetError,
    file_sha256,
    load_golden_set,
    load_meta,
    meta_path_for,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_SET = REPO_ROOT / "data" / "golden_set.jsonl"

KNOWN_GRADERS = {"numeric_exact_match", "llm_judge"}


@pytest.fixture(scope="module")
def items():
    return load_golden_set(GOLDEN_SET)


class TestCommittedGoldenSet:
    def test_loads_and_hash_matches_sidecar(self, items):
        """The load-time hash check passes on the committed artifact."""
        assert len(items) == 60

    def test_composition_matches_the_agreed_design(self, items):
        by_source = {}
        for item in items:
            by_source[item.source] = by_source.get(item.source, 0) + 1
        assert by_source == {"gsm8k": 50, "truthful_qa": 10}

    def test_every_grader_is_registered(self, items):
        """A typo'd grader name would silently leave items unscored."""
        assert {i.grader for i in items} <= KNOWN_GRADERS

    def test_gsm8k_items_use_the_deterministic_grader(self, items):
        """GSM8K having objective ground truth is what makes judge validation possible
        (design §8.3) -- if these ever route to the LLM judge, that capability is lost."""
        for item in items:
            if item.source == "gsm8k":
                assert item.grader == "numeric_exact_match"

    def test_gsm8k_reference_answers_are_numeric_and_normalized(self, items):
        import re

        numeric = re.compile(r"^-?\d+(?:\.\d+)?$")
        for item in items:
            if item.source == "gsm8k":
                assert numeric.match(item.reference_answer), f"{item.id}: {item.reference_answer!r}"
                assert "," not in item.reference_answer
                assert "$" not in item.reference_answer

    def test_gsm8k_rationales_have_calculator_annotations_stripped(self, items):
        for item in items:
            if item.source == "gsm8k":
                assert item.reference_rationale
                assert "<<" not in item.reference_rationale
                assert "####" not in item.reference_rationale

    def test_truthfulqa_native_category_preserved_without_colliding(self, items):
        """Our `category` is the coarse grouping; TruthfulQA's own lives in metadata (§7.4)."""
        for item in items:
            if item.source == "truthful_qa":
                assert item.category == "factual_qa"
                assert item.metadata.get("tqa_category")
                assert item.metadata.get("correct_answers")

    def test_ids_are_unique_and_stable_shaped(self, items):
        ids = [i.id for i in items]
        assert len(ids) == len(set(ids))
        for item in items:
            assert item.id.startswith(f"{item.source}-")
            assert item.id.split("-")[-1].isdigit()

    def test_no_required_field_is_blank(self, items):
        for item in items:
            for name in ("id", "source", "category", "question", "reference_answer", "grader"):
                assert getattr(item, name).strip()


class TestMetadataSidecar:
    def test_records_provenance_needed_to_reproduce_the_sample(self):
        meta = load_meta(GOLDEN_SET)
        assert meta["n_items"] == 60
        assert meta["seed"] == 42
        assert meta["sha256"] == file_sha256(GOLDEN_SET)
        for source in ("gsm8k", "truthful_qa"):
            prov = meta["sources"][source]
            assert prov["hf_id"] and prov["config"] and prov["split"]
            # Explicit indices make the sample reproducible even if upstream reshuffles.
            assert len(prov["indices"]) == meta["counts"][source]
            assert prov["indices"] == sorted(prov["indices"])

    def test_recorded_indices_agree_with_the_item_ids(self):
        """Cross-check: the sidecar and the data cannot drift apart silently."""
        meta = load_meta(GOLDEN_SET)
        items = load_golden_set(GOLDEN_SET)
        for source, prov in meta["sources"].items():
            from_ids = sorted(
                i.metadata["original_index"] for i in items if i.source == source
            )
            assert from_ids == prov["indices"]


class TestTamperDetection:
    """The gate's anti-gaming property, tested rather than asserted."""

    def _stage(self, tmp_path: Path) -> Path:
        target = tmp_path / "golden_set.jsonl"
        target.write_text(GOLDEN_SET.read_text(encoding="utf-8"), encoding="utf-8")
        meta_src = meta_path_for(GOLDEN_SET)
        meta_path_for(target).write_text(meta_src.read_text(encoding="utf-8"), encoding="utf-8")
        return target

    def test_dropping_a_hard_item_is_rejected(self, tmp_path):
        staged = self._stage(tmp_path)
        lines = staged.read_text(encoding="utf-8").splitlines(keepends=True)
        staged.write_text("".join(lines[:-1]), encoding="utf-8")
        with pytest.raises(GoldenSetError, match="hash mismatch"):
            load_golden_set(staged)

    def test_editing_a_reference_answer_is_rejected(self, tmp_path):
        staged = self._stage(tmp_path)
        lines = staged.read_text(encoding="utf-8").splitlines()
        row = json.loads(lines[0])
        row["reference_answer"] = "999999"
        lines[0] = json.dumps(row, ensure_ascii=False)
        staged.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(GoldenSetError, match="hash mismatch"):
            load_golden_set(staged)

    def test_verify_hash_false_is_an_explicit_opt_out(self, tmp_path):
        staged = self._stage(tmp_path)
        lines = staged.read_text(encoding="utf-8").splitlines(keepends=True)
        staged.write_text("".join(lines[:-1]), encoding="utf-8")
        assert len(load_golden_set(staged, verify_hash=False)) == 59

    def test_duplicate_ids_are_rejected(self, tmp_path):
        staged = self._stage(tmp_path)
        lines = staged.read_text(encoding="utf-8").splitlines(keepends=True)
        staged.write_text("".join(lines) + lines[0], encoding="utf-8")
        with pytest.raises(GoldenSetError, match="duplicate id"):
            load_golden_set(staged, verify_hash=False)

    def test_missing_required_field_is_rejected(self, tmp_path):
        staged = self._stage(tmp_path)
        row = json.loads(staged.read_text(encoding="utf-8").splitlines()[0])
        del row["reference_answer"]
        staged.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        with pytest.raises(GoldenSetError, match="reference_answer"):
            load_golden_set(staged, verify_hash=False)

    def test_helpful_error_when_file_is_absent(self, tmp_path):
        with pytest.raises(GoldenSetError, match="build_golden_set"):
            load_golden_set(tmp_path / "nope.jsonl")


class TestLimit:
    def test_limit_does_not_disable_the_hash_check(self):
        assert len(load_golden_set(GOLDEN_SET, limit=5)) == 5
