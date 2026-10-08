"""SQLite persistence for eval runs.

Stdlib only (design §4.1, §9).

Two properties matter more than anything else here:

* **Raw payloads are stored verbatim.** Inference costs minutes; metric arithmetic costs
  microseconds. Never re-run the expensive part to fix the cheap part (constraint C2).
* **Inserts are idempotent.** UNIQUE constraints plus INSERT OR IGNORE make `run` and `judge`
  resumable, so an interrupted 30-minute benchmark continues instead of restarting.

WAL mode is enabled so the dashboard can read while a run is still writing.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from viveka.core.types import Completion, Generation, Judgement, Timing

__all__ = ["Store", "generation_from_row", "new_run_id"]

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    started_utc      TEXT NOT NULL,
    finished_utc     TEXT,
    status           TEXT NOT NULL,           -- running | complete | failed
    viveka_version   TEXT,
    ollama_version   TEXT,
    git_sha          TEXT,
    config_json      TEXT,
    config_hash      TEXT,
    golden_set_hash  TEXT,
    prompt_version   TEXT,
    host_json        TEXT,
    notes            TEXT
);

CREATE TABLE IF NOT EXISTS generations (
    id               INTEGER PRIMARY KEY,
    run_id           TEXT NOT NULL REFERENCES runs(run_id),
    item_id          TEXT NOT NULL,
    model            TEXT NOT NULL,
    repeat_idx       INTEGER NOT NULL,
    is_warmup        INTEGER NOT NULL DEFAULT 0,
    output_text      TEXT,
    ttft_ms          REAL,
    total_ms         REAL,
    load_ms          REAL,
    prompt_eval_ms   REAL,
    decode_ms        REAL,
    prompt_tokens    INTEGER,
    output_tokens    INTEGER,
    done_reason      TEXT,
    prompt_version   TEXT,
    error            TEXT,
    raw_json         TEXT,
    created_utc      TEXT NOT NULL,
    UNIQUE (run_id, item_id, model, repeat_idx, is_warmup)
);

CREATE INDEX IF NOT EXISTS idx_generations_run   ON generations(run_id);
CREATE INDEX IF NOT EXISTS idx_generations_model ON generations(run_id, model);

CREATE TABLE IF NOT EXISTS judgements (
    id             INTEGER PRIMARY KEY,
    generation_id  INTEGER NOT NULL REFERENCES generations(id),
    grader_name    TEXT NOT NULL,
    grader_version TEXT NOT NULL,
    judge_model    TEXT,
    verdict        TEXT NOT NULL,
    score          REAL NOT NULL,
    confidence     REAL,
    reasoning      TEXT,
    latency_ms     REAL,
    error          TEXT,
    raw_json       TEXT,
    created_utc    TEXT NOT NULL,
    UNIQUE (generation_id, grader_name, grader_version)
);

CREATE INDEX IF NOT EXISTS idx_judgements_gen ON judgements(generation_id);

CREATE TABLE IF NOT EXISTS run_metrics (
    run_id       TEXT NOT NULL,
    model        TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    computed_utc TEXT NOT NULL,
    PRIMARY KEY (run_id, model)
);
"""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    """Sortable, human-readable run id: 20260730T172815Z-a1b2c3."""
    import uuid

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


class Store:
    """Thin, explicit data-access layer. No ORM: the schema is small and the queries are few."""

    def __init__(self, path: str | Path = "data/viveka.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")  # concurrent dashboard reads
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self.init_schema()

    # -- lifecycle ---------------------------------------------------------------------

    def init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- runs --------------------------------------------------------------------------

    def create_run(
        self,
        run_id: str,
        *,
        config: dict[str, Any] | None = None,
        config_hash: str | None = None,
        golden_set_hash: str | None = None,
        prompt_version: str | None = None,
        viveka_version: str | None = None,
        ollama_version: str | None = None,
        git_sha: str | None = None,
        host: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> None:
        """Register a run. Idempotent, so resuming an existing run_id is safe."""
        with self._conn:
            self._conn.execute(
                """INSERT OR IGNORE INTO runs
                   (run_id, started_utc, status, viveka_version, ollama_version, git_sha,
                    config_json, config_hash, golden_set_hash, prompt_version, host_json, notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    _utc_now(),
                    "running",
                    viveka_version,
                    ollama_version,
                    git_sha,
                    json.dumps(config or {}, sort_keys=True),
                    config_hash,
                    golden_set_hash,
                    prompt_version,
                    json.dumps(host or {}, sort_keys=True),
                    notes,
                ),
            )

    def finish_run(self, run_id: str, status: str = "complete") -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET status = ?, finished_utc = ? WHERE run_id = ?",
                (status, _utc_now(), run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def latest_run_id(self) -> str | None:
        row = self._conn.execute(
            "SELECT run_id FROM runs ORDER BY started_utc DESC, rowid DESC LIMIT 1"
        ).fetchone()
        return str(row["run_id"]) if row else None

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY started_utc DESC, rowid DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # -- generations -------------------------------------------------------------------

    def record_generation(self, gen: Generation, *, prompt_version: str | None = None) -> int:
        """Persist one generation and return its row id.

        INSERT OR IGNORE against the UNIQUE key, so re-running a stage is a no-op for work
        already done -- this is the resumability mechanism (design §9).
        """
        c = gen.completion
        t = c.timing
        with self._conn:
            cur = self._conn.execute(
                """INSERT OR IGNORE INTO generations
                   (run_id, item_id, model, repeat_idx, is_warmup, output_text,
                    ttft_ms, total_ms, load_ms, prompt_eval_ms, decode_ms,
                    prompt_tokens, output_tokens, done_reason, prompt_version,
                    error, raw_json, created_utc)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    gen.run_id,
                    gen.item_id,
                    gen.model,
                    gen.repeat_idx,
                    int(gen.is_warmup),
                    c.text,
                    t.ttft_ms,
                    t.total_ms,
                    t.load_ms,
                    t.prompt_eval_ms,
                    t.decode_ms,
                    t.prompt_tokens,
                    t.output_tokens,
                    c.raw.get("done_reason"),
                    prompt_version,
                    c.error,
                    json.dumps(c.raw, default=str),
                    _utc_now(),
                ),
            )
            if cur.lastrowid:
                return int(cur.lastrowid)

        # Already present (resumed run): look up the existing id.
        row = self._conn.execute(
            """SELECT id FROM generations
               WHERE run_id=? AND item_id=? AND model=? AND repeat_idx=? AND is_warmup=?""",
            (gen.run_id, gen.item_id, gen.model, gen.repeat_idx, int(gen.is_warmup)),
        ).fetchone()
        return int(row["id"])

    def completed_keys(self, run_id: str, model: str | None = None) -> set[tuple[str, int]]:
        """(item_id, repeat_idx) pairs already generated, so the runner can skip them.

        Warmups are excluded: a resumed run should re-warm, since the model is cold again.
        """
        sql = "SELECT item_id, repeat_idx FROM generations WHERE run_id=? AND is_warmup=0"
        params: list[Any] = [run_id]
        if model is not None:
            sql += " AND model=?"
            params.append(model)
        rows = self._conn.execute(sql, params).fetchall()
        return {(r["item_id"], r["repeat_idx"]) for r in rows}

    def generations(
        self, run_id: str, *, model: str | None = None, include_warmup: bool = False
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM generations WHERE run_id=?"
        params: list[Any] = [run_id]
        if not include_warmup:
            sql += " AND is_warmup=0"
        if model is not None:
            sql += " AND model=?"
            params.append(model)
        sql += " ORDER BY model, item_id, repeat_idx"
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def models_in_run(self, run_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT model FROM generations WHERE run_id=? ORDER BY model", (run_id,)
        ).fetchall()
        return [str(r["model"]) for r in rows]

    def iter_ungraded(
        self, run_id: str, grader_name: str, grader_version: str = "v1"
    ) -> Iterator[dict[str, Any]]:
        """Generations with no judgement yet from this grader -- makes `judge` resumable."""
        rows = self._conn.execute(
            """SELECT g.* FROM generations g
               LEFT JOIN judgements j
                 ON j.generation_id = g.id
                AND j.grader_name = ? AND j.grader_version = ?
               WHERE g.run_id = ? AND g.is_warmup = 0 AND j.id IS NULL
               ORDER BY g.model, g.item_id, g.repeat_idx""",
            (grader_name, grader_version, run_id),
        ).fetchall()
        yield from (dict(r) for r in rows)

    # -- judgements --------------------------------------------------------------------

    def record_judgement(self, j: Judgement) -> int:
        with self._conn:
            cur = self._conn.execute(
                """INSERT OR IGNORE INTO judgements
                   (generation_id, grader_name, grader_version, judge_model, verdict, score,
                    confidence, reasoning, latency_ms, error, raw_json, created_utc)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    j.generation_id,
                    j.grader_name,
                    j.grader_version,
                    j.judge_model,
                    j.verdict,
                    j.score,
                    j.confidence,
                    j.reasoning,
                    j.latency_ms,
                    j.error,
                    json.dumps(j.raw, default=str),
                    _utc_now(),
                ),
            )
            if cur.lastrowid:
                return int(cur.lastrowid)
        row = self._conn.execute(
            """SELECT id FROM judgements
               WHERE generation_id=? AND grader_name=? AND grader_version=?""",
            (j.generation_id, j.grader_name, j.grader_version),
        ).fetchone()
        return int(row["id"])

    def get_judgement(
        self, generation_id: int, grader_name: str, grader_version: str = "v1"
    ) -> dict[str, Any] | None:
        """Fetch an existing judgement, so expensive judge calls are never repeated."""
        row = self._conn.execute(
            """SELECT * FROM judgements
               WHERE generation_id=? AND grader_name=? AND grader_version=?""",
            (generation_id, grader_name, grader_version),
        ).fetchone()
        return dict(row) if row else None

    def judgements(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT j.*, g.item_id, g.model, g.repeat_idx
               FROM judgements j JOIN generations g ON g.id = j.generation_id
               WHERE g.run_id = ? ORDER BY g.model, g.item_id""",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- cached metrics ----------------------------------------------------------------

    def save_summary(self, run_id: str, model: str, summary: dict[str, Any]) -> None:
        """Cache a computed summary. Always recomputable; never the source of truth."""
        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO run_metrics (run_id, model, summary_json, computed_utc)
                   VALUES (?,?,?,?)""",
                (run_id, model, json.dumps(summary, default=str), _utc_now()),
            )

    def summaries(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT model, summary_json FROM run_metrics WHERE run_id=? ORDER BY model",
            (run_id,),
        ).fetchall()
        return [json.loads(r["summary_json"]) for r in rows]


def generation_from_row(row: dict[str, Any]) -> Generation:
    """Rehydrate a Generation from a database row (used by the judge and report stages)."""
    timing = Timing(
        total_ms=row["total_ms"] or 0.0,
        ttft_ms=row["ttft_ms"],
        load_ms=row["load_ms"],
        prompt_eval_ms=row["prompt_eval_ms"],
        decode_ms=row["decode_ms"],
        prompt_tokens=row["prompt_tokens"],
        output_tokens=row["output_tokens"],
    )
    raw: dict[str, Any] = {}
    if row.get("raw_json"):
        try:
            raw = json.loads(row["raw_json"])
        except json.JSONDecodeError:
            raw = {}
    return Generation(
        run_id=row["run_id"],
        item_id=row["item_id"],
        model=row["model"],
        repeat_idx=row["repeat_idx"],
        completion=Completion(
            text=row["output_text"] or "", timing=timing, raw=raw, error=row["error"]
        ),
        is_warmup=bool(row["is_warmup"]),
        id=row["id"],
    )
