#!/usr/bin/env python3
"""Build the frozen golden set from HuggingFace datasets.

Dev-time only -- requires the `data` extra (`pip install -e ".[dev,data]"`). The eval
runtime never imports `datasets`; it reads the committed JSONL this script produces.

    python scripts/build_golden_set.py                 # write data/golden_set.jsonl
    python scripts/build_golden_set.py --dry-run       # preview, write nothing

Determinism contract: the same --seed against the same upstream revisions produces
byte-identical JSONL, hence an identical SHA-256. That hash is recorded in the sidecar and
checked at load time, which is what stops the CI gate being gamed by quietly dropping hard
items (design §7.2).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

# --- source configuration -------------------------------------------------------------
# Canonical namespaced ids first, bare ids as fallback: `datasets` 5.x removed script-based
# loaders, and the bare aliases are the ones most likely to break next. Verified 2026-07-30
# that the namespaced ids resolve to Parquet conversions and load cleanly (design §7.4).
GSM8K_IDS = ("openai/gsm8k", "gsm8k")
GSM8K_CONFIG, GSM8K_SPLIT = "main", "test"

TRUTHFULQA_IDS = ("truthfulqa/truthful_qa", "truthful_qa")
TRUTHFULQA_CONFIG, TRUTHFULQA_SPLIT = "generation", "validation"

# GSM8K answers end with "#### <number>"; rationales carry <<48/2=24>> calculator annotations.
CALC_ANNOTATION_RE = re.compile(r"<<[^>]*>>")
NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")

BUILDER_VERSION = "1"


class BuildError(Exception):
    """Fatal: refuse to emit a golden set we cannot fully parse."""


def load_split(candidates: tuple[str, ...], config: str, split: str) -> tuple[str, Any]:
    from datasets import load_dataset

    errors = []
    for hf_id in candidates:
        try:
            return hf_id, load_dataset(hf_id, config, split=split)
        except Exception as exc:  # noqa: BLE001 - we genuinely want to try the next id
            errors.append(f"  {hf_id!r}: {type(exc).__name__}: {str(exc)[:200]}")
    raise BuildError(
        f"could not load {candidates[0]} ({config}/{split}). Tried:\n" + "\n".join(errors)
    )


def revision_of(hf_id: str) -> str | None:
    """Best-effort upstream commit sha, for provenance. Never fatal."""
    try:
        from huggingface_hub import dataset_info

        return str(dataset_info(hf_id).sha)
    except Exception:  # noqa: BLE001 - provenance is nice-to-have, not required
        return None


def sample_indices(n_rows: int, k: int, seed: int) -> list[int]:
    """Deterministic index sample.

    A fresh Random per source, rather than one shared stream, so changing the GSM8K count
    does not silently reshuffle the TruthfulQA selection.

    Chosen over datasets' own .shuffle(seed=) because explicit indices stay reproducible
    even if the library's shuffling internals change, and they can be recorded in the sidecar.
    """
    if k > n_rows:
        raise BuildError(f"requested {k} items but split has only {n_rows} rows")
    return sorted(random.Random(seed).sample(range(n_rows), k))


def parse_gsm8k_answer(answer: str) -> tuple[str, str]:
    """Split GSM8K's answer field into (rationale, final_answer).

    Raises BuildError rather than emitting an ungradeable item -- verified to never trigger
    on all 1319 upstream test rows, so it is a regression guard, not routine control flow.
    """
    if "####" not in answer:
        raise BuildError(f"no '####' marker in answer: {answer[:120]!r}")
    rationale, _, final = answer.rpartition("####")
    # Commas appear in ~1% of answers (e.g. "1,000"); '$' only ever in rationales, stripped
    # defensively. Both confirmed against the full split.
    final = final.strip().replace(",", "").replace("$", "")
    if not NUMERIC_RE.match(final):
        raise BuildError(f"final answer {final!r} is not numeric")
    return CALC_ANNOTATION_RE.sub("", rationale).strip(), final


def build_gsm8k(k: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hf_id, ds = load_split(GSM8K_IDS, GSM8K_CONFIG, GSM8K_SPLIT)
    indices = sample_indices(len(ds), k, seed)

    items = []
    for idx in indices:
        row = ds[idx]
        try:
            rationale, final = parse_gsm8k_answer(row["answer"])
        except BuildError as exc:
            raise BuildError(f"{hf_id} row {idx}: {exc}") from exc
        items.append(
            {
                "id": f"gsm8k-{GSM8K_SPLIT}-{idx:05d}",
                "source": "gsm8k",
                "category": "math_reasoning",
                "question": row["question"].strip(),
                "reference_answer": final,
                "reference_rationale": rationale,
                # Objective ground truth -> deterministic grader. These items are also what
                # make judge validation possible (design §8.3).
                "grader": "numeric_exact_match",
                "metadata": {"split": GSM8K_SPLIT, "original_index": idx},
            }
        )

    provenance = {
        "hf_id": hf_id,
        "config": GSM8K_CONFIG,
        "split": GSM8K_SPLIT,
        "split_rows": len(ds),
        "revision": revision_of(hf_id),
        "indices": indices,
    }
    return items, provenance


def build_truthfulqa(k: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hf_id, ds = load_split(TRUTHFULQA_IDS, TRUTHFULQA_CONFIG, TRUTHFULQA_SPLIT)
    indices = sample_indices(len(ds), k, seed)

    items = []
    for idx in indices:
        row = ds[idx]
        best = (row.get("best_answer") or "").strip()
        if not best:
            raise BuildError(f"{hf_id} row {idx}: empty best_answer")
        items.append(
            {
                "id": f"truthful_qa-{TRUTHFULQA_SPLIT}-{idx:05d}",
                "source": "truthful_qa",
                # Coarse grouping used by ModelSummary.by_category. TruthfulQA's own
                # fine-grained category ("Misconceptions", ...) goes to metadata to avoid
                # colliding with this field (design §7.4).
                "category": "factual_qa",
                "question": row["question"].strip(),
                "reference_answer": best,
                "grader": "llm_judge",
                "metadata": {
                    "split": TRUTHFULQA_SPLIT,
                    "original_index": idx,
                    "tqa_category": row.get("category"),
                    "tqa_type": row.get("type"),
                    # Richer judge context than best_answer alone: these enumerate what
                    # actually counts as right and wrong for this question.
                    "correct_answers": row.get("correct_answers") or [],
                    "incorrect_answers": row.get("incorrect_answers") or [],
                    "source_url": row.get("source"),
                },
            }
        )

    provenance = {
        "hf_id": hf_id,
        "config": TRUTHFULQA_CONFIG,
        "split": TRUTHFULQA_SPLIT,
        "split_rows": len(ds),
        "revision": revision_of(hf_id),
        "indices": indices,
    }
    return items, provenance


def render_jsonl(items: list[dict[str, Any]]) -> str:
    """Sorted by id and one compact object per line, so diffs are readable and stable."""
    lines = [
        json.dumps(item, ensure_ascii=False, sort_keys=False)
        for item in sorted(items, key=lambda i: i["id"])
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gsm8k-n", type=int, default=50, help="GSM8K items (default: 50)")
    ap.add_argument("--truthfulqa-n", type=int, default=10, help="TruthfulQA items (default: 10)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (default: 42)")
    ap.add_argument("--out", type=Path, default=Path("data/golden_set.jsonl"))
    ap.add_argument("--dry-run", action="store_true", help="preview without writing")
    args = ap.parse_args()

    print(f"Building golden set: {args.gsm8k_n} GSM8K + {args.truthfulqa_n} TruthfulQA, seed={args.seed}")

    gsm_items, gsm_prov = build_gsm8k(args.gsm8k_n, args.seed)
    print(f"  gsm8k       : {len(gsm_items):3d} items from {gsm_prov['hf_id']} ({gsm_prov['split_rows']} rows)")

    tqa_items, tqa_prov = build_truthfulqa(args.truthfulqa_n, args.seed)
    print(f"  truthful_qa : {len(tqa_items):3d} items from {tqa_prov['hf_id']} ({tqa_prov['split_rows']} rows)")

    items = gsm_items + tqa_items
    payload = render_jsonl(items)

    if args.dry_run:
        import hashlib

        print("\n--- first 2 items ---")
        for line in payload.splitlines()[:2]:
            print(json.dumps(json.loads(line), indent=2, ensure_ascii=False)[:700])
        print(f"\n--- would write {len(items)} items, sha256={hashlib.sha256(payload.encode()).hexdigest()} ---")
        print("DRY RUN: nothing written.")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="utf-8")

    # Hash the file as written -- this is the value load_golden_set() verifies.
    from viveka.core.datasets import file_sha256, meta_path_for

    sha = file_sha256(args.out)
    counts: dict[str, int] = {}
    for item in items:
        counts[item["source"]] = counts.get(item["source"], 0) + 1

    import datetime as _dt

    meta = {
        "builder_version": BUILDER_VERSION,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "sha256": sha,
        "n_items": len(items),
        "counts": counts,
        "seed": args.seed,
        "sources": {"gsm8k": gsm_prov, "truthful_qa": tqa_prov},
    }
    meta_file = meta_path_for(args.out)
    meta_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {args.out}  ({len(items)} items)")
    print(f"wrote {meta_file}")
    print(f"sha256: {sha}")

    # Close the loop: read it back through the real loader, including the hash check.
    from viveka.core.datasets import load_golden_set

    loaded = load_golden_set(args.out)
    print(f"verified: loader accepted {len(loaded)} items, hash matches")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BuildError as exc:
        print(f"\nBUILD FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
