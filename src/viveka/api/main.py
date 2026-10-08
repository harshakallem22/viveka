"""FastAPI backend serving eval results to the dashboard.

Optional extra: `pip install viveka[api]`. Nothing in `viveka.core` imports this module -- the
dependency runs one way only, enforced by tests/test_import_boundary.py (constraint C1).

Read-only by design. The API never triggers a benchmark: inference takes tens of minutes and
belongs in an explicit CLI invocation, not behind an HTTP request that a browser might retry. That
also means this can be deployed against a committed demo database with no Ollama present at all,
which is what makes the portfolio demo hostable (design §12).

"Live" updates are polling, not WebSockets. The runner writes generations incrementally in WAL
mode, so a poll every couple of seconds shows progress during a run -- an order of magnitude less
machinery than a socket for data that changes every few seconds.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from viveka import __version__
from viveka.core.agreement import compare_verdicts
from viveka.core.datasets import GoldenSetError, load_golden_set
from viveka.core.metrics import summarize
from viveka.core.store import Store, generation_from_row
from viveka.core.types import Judgement

#: Overridable so the hosted demo can point at a committed read-only database.
DB_PATH = os.environ.get("VIVEKA_DB", "data/viveka.db")
GOLDEN_SET_PATH = os.environ.get("VIVEKA_GOLDEN_SET", "data/golden_set.jsonl")
DASHBOARD_DIST = Path(os.environ.get("VIVEKA_DASHBOARD", "dashboard/dist"))

app = FastAPI(
    title="Viveka",
    version=__version__,
    description="LLM eval harness — leaderboard, latency percentiles, judge reliability.",
)

# Vite's dev server runs on another port, so the dashboard needs CORS during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _store() -> Store:
    """A fresh connection per request: sqlite3 connections are not thread-safe to share."""
    return Store(DB_PATH)


def _golden_items() -> dict[str, Any]:
    try:
        # Hash verification is skipped here: the API must still serve a historical run whose eval
        # set has since moved on. Integrity enforcement belongs to the CLI and the CI gate.
        return {i.id: i for i in load_golden_set(GOLDEN_SET_PATH, verify_hash=False)}
    except GoldenSetError:
        return {}


def _resolve_run(store: Store, run_id: str) -> str:
    resolved = store.latest_run_id() if run_id == "latest" else run_id
    if not resolved or not store.get_run(resolved):
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    return resolved


def _typed_judgements(rows: list[dict[str, Any]]) -> list[Judgement]:
    return [
        Judgement(
            generation_id=r["generation_id"],
            grader_name=r["grader_name"],
            grader_version=r["grader_version"],
            judge_model=r["judge_model"],
            verdict=r["verdict"],
            score=r["score"],
            confidence=r["confidence"],
            reasoning=r["reasoning"],
            latency_ms=r["latency_ms"],
            error=r["error"],
        )
        for r in rows
    ]


@app.get("/api/healthz")
def healthz() -> dict[str, Any]:
    store = _store()
    try:
        runs = store.list_runs(limit=1)
        return {"status": "ok", "version": __version__, "db": DB_PATH, "runs": len(runs)}
    finally:
        store.close()


@app.get("/api/runs")
def list_runs(limit: int = Query(default=50, ge=1, le=500)) -> list[dict[str, Any]]:
    """Run index, newest first."""
    store = _store()
    try:
        out = []
        for run in store.list_runs(limit=limit):
            generations = store.generations(run["run_id"])
            out.append(
                {
                    "run_id": run["run_id"],
                    "started_utc": run["started_utc"],
                    "finished_utc": run["finished_utc"],
                    "status": run["status"],
                    "prompt_version": run["prompt_version"],
                    "golden_set_hash": run["golden_set_hash"],
                    "viveka_version": run["viveka_version"],
                    "ollama_version": run["ollama_version"],
                    "models": store.models_in_run(run["run_id"]),
                    "n_generations": len(generations),
                    "n_errors": sum(1 for g in generations if g["error"]),
                }
            )
        return out
    finally:
        store.close()


@app.get("/api/runs/{run_id}/summary")
def run_summary(
    run_id: str,
    grader: str | None = Query(
        default="numeric_exact_match",
        description="Restrict accuracy to one grader; required when several graders scored the "
        "same generations, or accuracy double-counts.",
    ),
) -> dict[str, Any]:
    """The leaderboard: per-model accuracy with CI, latency percentiles, throughput."""
    store = _store()
    try:
        rid = _resolve_run(store, run_id)
        meta = store.get_run(rid)
        assert meta is not None
        items = _golden_items()
        judgements = _typed_judgements(store.judgements(rid))

        models = []
        for model in store.models_in_run(rid):
            generations = [
                generation_from_row(r)
                for r in store.generations(rid, model=model, include_warmup=True)
            ]
            summary = summarize(
                model,
                generations,
                judgements=judgements or None,
                items=items,
                grader_name=grader,
            )
            models.append(summary.as_dict())

        return {
            "run_id": rid,
            "status": meta["status"],
            "started_utc": meta["started_utc"],
            "finished_utc": meta["finished_utc"],
            "prompt_version": meta["prompt_version"],
            "golden_set_hash": meta["golden_set_hash"],
            "ollama_version": meta["ollama_version"],
            "viveka_version": meta["viveka_version"],
            "grader": grader,
            "models": models,
        }
    finally:
        store.close()


@app.get("/api/runs/{run_id}/generations")
def run_generations(
    run_id: str,
    model: str | None = None,
    limit: int = Query(default=500, ge=1, le=5000),
) -> list[dict[str, Any]]:
    """Per-item drill-down, including the judge's reasoning.

    The most persuasive view in the dashboard: it shows the eval is inspectable rather than a
    black box that emits a number.
    """
    store = _store()
    try:
        rid = _resolve_run(store, run_id)
        items = _golden_items()
        verdicts: dict[int, list[dict[str, Any]]] = {}
        for j in store.judgements(rid):
            verdicts.setdefault(j["generation_id"], []).append(
                {
                    "grader": j["grader_name"],
                    "verdict": j["verdict"],
                    "score": j["score"],
                    "confidence": j["confidence"],
                    "reasoning": j["reasoning"],
                    "judge_model": j["judge_model"],
                }
            )

        out = []
        for row in store.generations(rid, model=model)[:limit]:
            item = items.get(row["item_id"])
            out.append(
                {
                    "id": row["id"],
                    "item_id": row["item_id"],
                    "model": row["model"],
                    "category": item.category if item else None,
                    "question": item.question if item else None,
                    "reference_answer": item.reference_answer if item else None,
                    "output_text": row["output_text"],
                    "ttft_ms": row["ttft_ms"],
                    "total_ms": row["total_ms"],
                    "decode_ms": row["decode_ms"],
                    "output_tokens": row["output_tokens"],
                    "done_reason": row["done_reason"],
                    "truncated": row["done_reason"] == "length",
                    "error": row["error"],
                    "judgements": verdicts.get(row["id"], []),
                }
            )
        return out
    finally:
        store.close()


@app.get("/api/runs/{run_id}/latencies")
def run_latencies(run_id: str) -> dict[str, list[float]]:
    """Raw per-model latency samples, for distribution charts.

    Percentiles alone hide shape -- llama3.1:8b's tail turned out to be two runaway generations,
    which a P50/P95 pair cannot show but a distribution can.
    """
    store = _store()
    try:
        rid = _resolve_run(store, run_id)
        out: dict[str, list[float]] = {}
        for model in store.models_in_run(rid):
            out[model] = [
                r["total_ms"]
                for r in store.generations(rid, model=model)
                if r["total_ms"] and not r["error"]
            ]
        return out
    finally:
        store.close()


@app.get("/api/runs/{run_id}/agreement")
def run_agreement(
    run_id: str,
    truth_grader: str = "numeric_exact_match",
    judge_grader: str = "llm_judge",
) -> dict[str, Any]:
    """Judge reliability: Cohen's kappa and the confusion matrix, overall and per contestant.

    Only items scored by *both* graders are comparable, which in practice means the ones with
    objective ground truth.
    """
    store = _store()
    try:
        rid = _resolve_run(store, run_id)
        truth: dict[int, str] = {}
        judged: dict[int, str] = {}
        model_of: dict[int, str] = {}
        for j in store.judgements(rid):
            model_of[j["generation_id"]] = j["model"]
            if j["grader_name"] == truth_grader:
                truth[j["generation_id"]] = j["verdict"]
            elif j["grader_name"] == judge_grader:
                judged[j["generation_id"]] = j["verdict"]

        shared = sorted(set(truth) & set(judged))
        pairs = [(truth[g], judged[g]) for g in shared]
        by_model: dict[str, list[tuple[str, str]]] = {}
        for g in shared:
            by_model.setdefault(model_of[g], []).append((truth[g], judged[g]))

        return {
            "run_id": rid,
            "truth_grader": truth_grader,
            "judge_grader": judge_grader,
            "overall": compare_verdicts(pairs).as_dict(),
            "by_model": {m: compare_verdicts(p).as_dict() for m, p in sorted(by_model.items())},
        }
    finally:
        store.close()


# --- static dashboard ---------------------------------------------------------------------
# Mounted last so it cannot shadow /api. Absent in development, where Vite serves the frontend.
if DASHBOARD_DIST.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=DASHBOARD_DIST / "assets"),
        name="assets",
    )

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(DASHBOARD_DIST / "index.html")
