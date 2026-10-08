#!/usr/bin/env python3
"""Build a committed, read-only demo database for the hosted dashboard.

    python scripts/make_demo_db.py --run latest

The hosted demo cannot run Ollama -- no GPU, and a benchmark takes tens of minutes. So instead of
faking data or apologising for an empty page, the demo ships a real run: one complete benchmark
extracted from the local database, with everything else pruned and the file VACUUMed.

The result is a portfolio demo that serves genuine measured numbers with zero inference at request
time (design §12).
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

from viveka.core.store import Store


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="latest", help="run id to keep, or 'latest'")
    ap.add_argument("--source", default="data/viveka.db")
    ap.add_argument("--out", type=Path, default=Path("data/demo.db"))
    args = ap.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"no source database at {source}", file=sys.stderr)
        return 1

    store = Store(source)
    rid = store.latest_run_id() if args.run == "latest" else args.run
    meta = store.get_run(rid) if rid else None
    store.close()
    if not rid or meta is None:
        print(f"no such run: {args.run}", file=sys.stderr)
        return 1
    if meta["status"] != "complete":
        print(f"refusing to ship run {rid}: status is {meta['status']!r}, not 'complete'", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, args.out)

    conn = sqlite3.connect(args.out)
    try:
        # Order matters: judgements reference generations, which reference runs.
        conn.execute(
            "DELETE FROM judgements WHERE generation_id IN "
            "(SELECT id FROM generations WHERE run_id != ?)",
            (rid,),
        )
        conn.execute("DELETE FROM generations WHERE run_id != ?", (rid,))
        conn.execute("DELETE FROM run_metrics WHERE run_id != ?", (rid,))
        conn.execute("DELETE FROM runs WHERE run_id != ?", (rid,))
        conn.commit()
        # WAL files must not be shipped alongside; a plain journal keeps the artifact a single file.
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("VACUUM")
        conn.commit()
    finally:
        conn.close()

    verify = Store(args.out)
    try:
        runs = verify.list_runs()
        generations = verify.generations(rid)
        judgements = verify.judgements(rid)
    finally:
        verify.close()

    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out}  ({size_kb:.0f} KB)")
    print(f"  runs        : {len(runs)} ({rid})")
    print(f"  generations : {len(generations)}")
    print(f"  judgements  : {len(judgements)}")
    print(f"  models      : {', '.join(verify_models := sorted({g['model'] for g in generations}))}")
    print()
    print("Serve it with:  VIVEKA_DB=data/demo.db viveka serve")
    return 0 if verify_models else 1


if __name__ == "__main__":
    sys.exit(main())
