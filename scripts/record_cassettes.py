#!/usr/bin/env python3
"""Record CI cassettes from a real benchmark run already stored in SQLite.

    python scripts/record_cassettes.py --run latest

No new model calls are made: everything is derived from generations and judgements already
persisted, which is the payoff of storing raw payloads verbatim (constraint C2). The result is
fixtures containing genuine model output, so Tier-1 CI exercises the real pipeline against real
text rather than against what a fixture author imagined a model would say.

Prompts are recomputed with the live templates, so if a template changes the cassettes must be
re-recorded -- which is the intended signal, not an inconvenience.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from viveka.core.datasets import load_golden_set
from viveka.core.judges import build_judge_prompt
from viveka.core.prompts import build_prompt
from viveka.core.providers.replay import prompt_key
from viveka.core.store import Store

DEFAULT_OUT = Path("tests/fixtures/cassettes")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="latest", help="run id, or 'latest'")
    ap.add_argument("--db", default="data/viveka.db")
    ap.add_argument("--golden-set", default="data/golden_set.jsonl")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--gsm8k-n", type=int, default=8, help="GSM8K items to record per model")
    ap.add_argument("--truthfulqa-n", type=int, default=3, help="TruthfulQA items per model")
    args = ap.parse_args()

    store = Store(args.db)
    rid = store.latest_run_id() if args.run == "latest" else args.run
    if not rid or not store.get_run(rid):
        print(f"no such run: {args.run}", file=sys.stderr)
        return 1

    items = {i.id: i for i in load_golden_set(args.golden_set)}

    # A deterministic, grader-diverse slice: both graders must be exercised, or the replayed
    # pipeline would only ever test one scoring path.
    gsm = sorted(i for i, it in items.items() if it.source == "gsm8k")[: args.gsm8k_n]
    tqa = sorted(i for i, it in items.items() if it.source == "truthful_qa")[: args.truthfulqa_n]
    chosen = set(gsm + tqa)
    print(f"run {rid}: recording {len(gsm)} GSM8K + {len(tqa)} TruthfulQA items per model")

    args.out.mkdir(parents=True, exist_ok=True)
    for stale in args.out.glob("*.json"):
        stale.unlink()

    judge_entries: dict[str, dict] = {}
    judge_model: str | None = None
    total = 0

    for model in store.models_in_run(rid):
        entries: dict[str, dict] = {}
        for row in store.generations(rid, model=model):
            if row["item_id"] not in chosen:
                continue
            item = items[row["item_id"]]
            entries[prompt_key(build_prompt(item))] = {
                "item_id": row["item_id"],
                "text": row["output_text"] or "",
                "ttft_ms": row["ttft_ms"],
                "total_ms": row["total_ms"],
                "load_ms": row["load_ms"],
                "prompt_eval_ms": row["prompt_eval_ms"],
                "decode_ms": row["decode_ms"],
                "prompt_tokens": row["prompt_tokens"],
                "output_tokens": row["output_tokens"],
                "done_reason": row["done_reason"],
                "error": row["error"],
            }

            # Judge cassette: reconstruct the judge's own response from the stored verdict, so the
            # LLMJudge parsing path is exercised too.
            judged = store.get_judgement(row["id"], "llm_judge")
            if judged:
                judge_model = judged["judge_model"]
                payload = {
                    "verdict": judged["verdict"],
                    "confidence": judged["confidence"]
                    if judged["confidence"] is not None
                    else 1.0,
                    "reasoning": judged["reasoning"] or "recorded verdict",
                }
                judge_prompt = build_judge_prompt(item, row["output_text"] or "")
                judge_entries[prompt_key(judge_prompt)] = {
                    "item_id": row["item_id"],
                    "text": json.dumps(payload),
                    "total_ms": judged["latency_ms"] or 5400.0,
                    "ttft_ms": 400.0,
                    "decode_ms": 5000.0,
                    "output_tokens": 76,
                    "done_reason": "stop",
                }

        slug = model.replace(":", "_").replace("/", "_")
        path = args.out / f"{slug}.json"
        path.write_text(
            json.dumps({"model": model, "entries": entries}, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        total += len(entries)
        print(f"  wrote {path}  ({len(entries)} entries)")

    if judge_entries:
        (args.out / "judge.json").write_text(
            json.dumps(
                {"model": judge_model or "ollama:qwen3:8b", "entries": judge_entries},
                indent=1,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {args.out / 'judge.json'}  ({len(judge_entries)} entries)")

    print(f"\n{total} contestant entries recorded. These are real model outputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
