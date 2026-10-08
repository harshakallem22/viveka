"""Viveka command line.

Stages are separate commands on purpose (design §5.4): inference costs minutes, so a bug in
the judge prompt must cost one `viveka judge` re-run, not a full re-generation.

    viveka run      generate outputs           (this milestone)
    viveka judge    score them                 (Milestone 3)
    viveka report   aggregate metrics          (Milestone 2)
    viveka gate     compare against baseline   (Milestone 5)

All presentation lives here; `viveka.core` emits events and data.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from viveka.config import DEFAULT_CONFIG_PATH, Config, load_config
from viveka.core.agreement import compare_verdicts
from viveka.core.datasets import GoldenSetError, file_sha256, load_golden_set
from viveka.core.gate import evaluate_gate, summarize_markdown, thresholds_from_mapping
from viveka.core.graders import get_grader
from viveka.core.judges import JUDGE_SCHEMA, LLMJudge
from viveka.core.metrics import ModelSummary, summarize
from viveka.core.providers.ollama import OllamaProvider, ollama_version
from viveka.core.runner import RunConfig, Runner, RunProgress
from viveka.core.store import Store, generation_from_row
from viveka.core.types import Judgement

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="LLM eval harness: LLM-as-judge scoring, P50/P95 + TTFT latency, CI regression gating.",
)
console = Console()


def _build_providers(cfg: Config, only: list[str] | None) -> list[OllamaProvider]:
    contestants = cfg.contestants
    if only:
        wanted = {m if ":" in m else f"{m}:latest" for m in only}
        contestants = [c for c in contestants if c.model in wanted]
        missing = wanted - {c.model for c in contestants}
        if missing:
            raise typer.BadParameter(
                f"not in viveka.yaml contestants: {sorted(missing)}"
            )
    if not contestants:
        raise typer.BadParameter("no contestants selected; add some to viveka.yaml")

    return [
        OllamaProvider(
            c.model,
            base_url=cfg.run.base_url,
            timeout_s=cfg.run.timeout_s,
            keep_alive=cfg.run.keep_alive,
            options={**cfg.run.defaults.model_dump(), **c.options},
        )
        for c in contestants
    ]


@app.command()
def run(
    config_path: Annotated[
        Path, typer.Option("--config", "-c", help="Path to viveka.yaml")
    ] = Path(DEFAULT_CONFIG_PATH),
    models: Annotated[
        list[str] | None,
        typer.Option("--model", "-m", help="Only these models (repeatable)"),
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", "-n", help="Use only the first N eval items")
    ] = None,
    repeats: Annotated[
        int | None, typer.Option("--repeats", "-r", help="Override run.repeats")
    ] = None,
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Resume/extend an existing run")
    ] = None,
    no_warmup: Annotated[
        bool, typer.Option("--no-warmup", help="Skip warmup (will distort P95 -- for debugging)")
    ] = False,
) -> None:
    """Generate model outputs for the golden set, capturing TTFT, latency and throughput."""
    cfg = load_config(config_path)
    if repeats is not None:
        cfg.run.repeats = repeats

    items = load_golden_set(cfg.run.golden_set, limit=limit)
    providers = _build_providers(cfg, models)

    server = ollama_version(cfg.run.base_url)
    if server is None:
        console.print(
            f"[red]Cannot reach Ollama at {cfg.run.base_url}.[/red] Is the server running?"
        )
        raise typer.Exit(code=1)

    if not cfg.judge_is_a_contestant():
        judge_note = f"judge {cfg.judge.model} (not a contestant [green]ok[/green])"
    else:
        judge_note = f"[yellow]judge {cfg.judge.model} is also a contestant: self-bias risk[/yellow]"

    console.print(
        f"[bold]Viveka run[/bold]  ollama {server}  |  {len(items)} items x "
        f"{len(providers)} models x {cfg.run.repeats} repeat(s) = "
        f"[bold]{len(items) * len(providers) * cfg.run.repeats}[/bold] generations"
    )
    console.print(f"  {judge_note}")
    if no_warmup:
        console.print("  [yellow]warmup disabled: latency percentiles will include cold start[/yellow]")

    truncated: list[str] = []
    failures: list[str] = []

    def on_progress(ev: RunProgress) -> None:
        if ev.is_warmup:
            cold = ev.completion.timing.load_ms if ev.completion else None
            console.print(
                f"  [dim]warmup {ev.model}: load "
                f"{cold / 1000:.2f}s[/dim]" if cold else f"  [dim]warmup {ev.model}[/dim]"
            )
            return
        if ev.skipped:
            return

        c = ev.completion
        assert c is not None
        prefix = f"  [{ev.model_index}/{ev.model_total}] {ev.model} {ev.item_index}/{ev.item_total} {ev.item_id}"
        if not c.ok:
            failures.append(f"{ev.model}/{ev.item_id}: {c.error}")
            console.print(f"{prefix} [red]FAILED[/red] {c.error}")
            return
        if c.raw.get("done_reason") == "length":
            truncated.append(f"{ev.model}/{ev.item_id}")
        ttft = f"{c.timing.ttft_ms:.0f}ms" if c.timing.ttft_ms else "n/a"
        tps = f"{c.timing.decode_tps:.1f}" if c.timing.decode_tps else "n/a"
        console.print(
            f"{prefix} [green]ok[/green] "
            f"ttft={ttft} total={c.timing.total_ms / 1000:.2f}s tok/s={tps} "
            f"out={c.timing.output_tokens}"
        )

    runner = Runner(
        Store(cfg.run.db),
        config=RunConfig(
            repeats=cfg.run.repeats,
            warmup=not no_warmup,
            resume=cfg.run.resume,
            defaults=cfg.run.defaults.model_dump(),
        ),
        on_progress=on_progress,
    )

    try:
        rid = runner.run(
            items,
            providers,
            run_id=run_id,
            golden_set_hash=file_sha256(cfg.run.golden_set),
            config_snapshot=cfg.snapshot(),
            ollama_version=server,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Rows already written are preserved;[/yellow]")
        console.print("[yellow]resume with --run-id <id>.[/yellow]")
        raise typer.Exit(code=130) from None
    finally:
        for p in providers:
            p.close()

    console.print(f"\n[bold green]Run complete:[/bold green] {rid}")
    if truncated:
        console.print(
            f"[yellow]{len(truncated)} response(s) hit the token cap[/yellow] "
            f"(done_reason=length). These may score incorrect for a harness reason, not a "
            f"model reason -- consider raising run.defaults.num_predict."
        )
        for t in truncated[:5]:
            console.print(f"  [dim]{t}[/dim]")
    if failures:
        console.print(f"[red]{len(failures)} request(s) failed[/red]")
        for f in failures[:5]:
            console.print(f"  [dim]{f}[/dim]")
    console.print(f"[dim]Stored in {cfg.run.db}. Next: viveka report {rid}[/dim]")


@app.command()
def runs(
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path(DEFAULT_CONFIG_PATH),
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
) -> None:
    """List recorded runs."""
    cfg = load_config(config_path)
    store = Store(cfg.run.db)
    rows = store.list_runs(limit=limit)
    if not rows:
        console.print("No runs yet. Try: viveka run --limit 5")
        return

    table = Table(title="Runs", header_style="bold")
    for col in ("run_id", "started", "status", "models", "generations", "prompt"):
        table.add_column(col)
    for r in rows:
        gens = store.generations(r["run_id"])
        table.add_row(
            r["run_id"],
            (r["started_utc"] or "")[:19],
            r["status"],
            str(len(store.models_in_run(r["run_id"]))),
            str(len(gens)),
            r["prompt_version"] or "-",
        )
    console.print(table)




def _judge_provider(cfg: Config) -> OllamaProvider:
    """The judge model, configured for deterministic schema-constrained grading."""
    return OllamaProvider(
        cfg.judge.model,
        base_url=cfg.run.base_url,
        timeout_s=cfg.run.timeout_s,
        keep_alive=cfg.run.keep_alive,
        options={
            "temperature": cfg.judge.temperature,
            "seed": cfg.judge.seed,
            "num_predict": cfg.judge.num_predict,
        },
        think=cfg.judge.think,
        response_format=JUDGE_SCHEMA,
    )


@app.command()
def judge(
    run_id: Annotated[str, typer.Argument(help="Run id, or 'latest'")] = "latest",
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path(DEFAULT_CONFIG_PATH),
    only: Annotated[
        str | None, typer.Option("--grader", help="Only run this grader")
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", "-n", help="Grade at most N generations per grader")
    ] = None,
) -> None:
    """Score a run's generations using each item's designated grader.

    Resumable: only generations without a judgement from that grader are graded, so a crash
    partway through does not force a re-judge of everything.
    """
    cfg = load_config(config_path)
    store = Store(cfg.run.db)
    rid = store.latest_run_id() if run_id == "latest" else run_id
    if not rid or not store.get_run(rid):
        console.print(f"[red]No such run:[/red] {run_id}")
        raise typer.Exit(code=1)

    items = {i.id: i for i in load_golden_set(cfg.run.golden_set)}

    # Deterministic graders first, then the LLM judge. This ordering is not cosmetic: it keeps
    # all judge-model calls in one contiguous block so Ollama loads qwen3 once rather than
    # thrashing 5GB of weights (design §1.4).
    needed = {items[i].grader for i in items}
    if only:
        needed = {g for g in needed if g == only} or {only}

    judge_provider: OllamaProvider | None = None
    graders: dict[str, object] = {}
    for name in sorted(needed, key=lambda n: (n == "llm_judge", n)):
        if name == "llm_judge":
            judge_provider = _judge_provider(cfg)
            graders[name] = LLMJudge(judge_provider, version=cfg.judge.prompt_version)
        else:
            graders[name] = get_grader(name)

    console.print(f"[bold]Judging[/bold] {rid} with: {', '.join(graders)}")

    try:
        for name, grader in graders.items():
            version = grader.version  # type: ignore[attr-defined]
            # An item is graded only by the grader its golden-set entry designates (C5).
            pending = [
                r
                for r in store.iter_ungraded(rid, name, version)
                if (it := items.get(r["item_id"])) is not None and it.grader == name
            ]
            if limit:
                pending = pending[:limit]
            if not pending:
                console.print(f"  [dim]{name}: nothing to do[/dim]")
                continue

            if isinstance(grader, LLMJudge):
                console.print(f"  {name}: {len(pending)} to grade via {grader.judge_model}")
                grader.provider.warmup()
            else:
                console.print(f"  {name}: {len(pending)} to grade")

            counts = {"correct": 0, "incorrect": 0, "error": 0}
            for row in pending:
                item = items[row["item_id"]]
                score = grader.grade(  # type: ignore[attr-defined]
                    item,
                    row["output_text"] or "",
                    truncated=row["done_reason"] == "length",
                )
                counts[score.verdict] += 1
                store.record_judgement(
                    score.to_judgement(
                        row["id"],
                        grader_name=name,
                        grader_version=version,
                        judge_model=getattr(grader, "judge_model", None),
                    )
                )
            console.print(
                f"    correct={counts['correct']} incorrect={counts['incorrect']} "
                f"[yellow]error={counts['error']}[/yellow]"
            )
    finally:
        if judge_provider is not None:
            judge_provider.unload()
            judge_provider.close()

    console.print(f"\n[bold green]Judged.[/bold green] Next: viveka report {rid}")


@app.command(name="validate-judge")
def validate_judge(
    run_id: Annotated[str, typer.Argument(help="Run id, or 'latest'")] = "latest",
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path(DEFAULT_CONFIG_PATH),
    limit: Annotated[
        int | None, typer.Option("--limit", "-n", help="Validate on at most N generations")
    ] = None,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write the agreement report here")
    ] = None,
) -> None:
    """Measure how well the LLM judge agrees with objective ground truth.

    Runs the LLM judge over items that also have a deterministic grader (GSM8K), then reports raw
    agreement, Cohen's kappa, and the confusion matrix. This is what makes the LLM-as-judge
    approach defensible rather than assumed (design §8.3).
    """
    cfg = load_config(config_path)
    store = Store(cfg.run.db)
    rid = store.latest_run_id() if run_id == "latest" else run_id
    if not rid or not store.get_run(rid):
        console.print(f"[red]No such run:[/red] {run_id}")
        raise typer.Exit(code=1)

    items = {i.id: i for i in load_golden_set(cfg.run.golden_set)}
    # Only items with objective ground truth can validate a judge.
    objective = {k: v for k, v in items.items() if v.grader == "numeric_exact_match"}
    if not objective:
        console.print("[red]No items with a deterministic grader; cannot validate.[/red]")
        raise typer.Exit(code=1)

    rows = [r for r in store.generations(rid) if r["item_id"] in objective]
    if limit:
        rows = rows[:limit]
    if not rows:
        console.print(f"[yellow]No generations to validate in {rid}.[/yellow]")
        raise typer.Exit(code=1)

    truth_grader = get_grader("numeric_exact_match")
    provider = _judge_provider(cfg)
    judge_grader = LLMJudge(provider, version=cfg.judge.prompt_version)

    console.print(
        f"[bold]Validating judge[/bold] {cfg.judge.model} against "
        f"{truth_grader.name} on {len(rows)} generations from {rid}"
    )
    console.print("[dim]Each item needs one judge call; this is the slow part.[/dim]")

    pairs: list[tuple[str, str]] = []
    per_model: dict[str, list[tuple[str, str]]] = {}
    disagreements: list[tuple[str, str, str, str]] = []
    reused = 0

    try:
        provider.warmup()
        for n, row in enumerate(rows, start=1):
            item = objective[row["item_id"]]
            text = row["output_text"] or ""

            truth = truth_grader.grade(item, text, truncated=row["done_reason"] == "length")
            store.record_judgement(
                truth.to_judgement(
                    row["id"],
                    grader_name=truth_grader.name,
                    grader_version=truth_grader.version,
                )
            )

            # Reuse a stored judge verdict when we already paid for it: each judge call costs
            # seconds, so re-running validation must not re-spend the whole budget.
            existing = store.get_judgement(row["id"], judge_grader.name, judge_grader.version)
            if existing is not None:
                judge_verdict = str(existing["verdict"])
                judge_reason = existing["reasoning"] or ""
                reused += 1
            else:
                score = judge_grader.grade(item, text)
                store.record_judgement(
                    score.to_judgement(
                        row["id"],
                        grader_name=judge_grader.name,
                        grader_version=judge_grader.version,
                        judge_model=judge_grader.judge_model,
                    )
                )
                judge_verdict = score.verdict
                judge_reason = score.reasoning or ""

            pairs.append((truth.verdict, judge_verdict))
            per_model.setdefault(row["model"], []).append((truth.verdict, judge_verdict))
            if (
                truth.verdict in ("correct", "incorrect")
                and judge_verdict in ("correct", "incorrect")
                and truth.verdict != judge_verdict
            ):
                disagreements.append(
                    (row["model"], row["item_id"], truth.verdict, judge_reason)
                )
            if n % 10 == 0:
                console.print(f"  [dim]{n}/{len(rows)}[/dim]")
    finally:
        provider.unload()
        provider.close()

    if reused:
        console.print(f"[dim]Reused {reused} previously stored judge verdict(s).[/dim]")

    report = compare_verdicts(pairs)

    console.print(f"\n[bold]Judge reliability[/bold] — {cfg.judge.model} vs objective ground truth")
    table = Table(header_style="bold")
    table.add_column("metric", justify="left")
    table.add_column("value", justify="right")
    table.add_row("comparable pairs", str(report.n))
    table.add_row("skipped (unextractable)", str(report.n_skipped))
    table.add_row("raw agreement", f"{report.raw_agreement:.1%}")
    table.add_row("Cohen's kappa", f"{report.cohen_kappa:.3f} ({report.interpretation})")
    table.add_row(
        "false positives", f"{report.false_positive}"
        + (f"  ({report.false_positive_rate:.1%} of wrong answers)" if report.false_positive_rate is not None else "")
    )
    table.add_row(
        "false negatives", f"{report.false_negative}"
        + (f"  ({report.false_negative_rate:.1%} of right answers)" if report.false_negative_rate is not None else "")
    )
    console.print(table)

    matrix = Table(title="Confusion matrix", header_style="bold", title_style="bold")
    matrix.add_column("", justify="left")
    matrix.add_column("judge: correct", justify="right")
    matrix.add_column("judge: incorrect", justify="right")
    matrix.add_row("truth: correct", str(report.true_positive), str(report.false_negative))
    matrix.add_row("truth: incorrect", f"[red]{report.false_positive}[/red]", str(report.true_negative))
    console.print(matrix)

    if report.truth_positive_rate is not None and not 0.2 <= report.truth_positive_rate <= 0.8:
        console.print(
            f"[yellow]Class balance is skewed ({report.truth_positive_rate:.0%} of answers are "
            f"correct), so raw agreement is inflated — quote kappa, not agreement.[/yellow]"
        )
    if report.false_positive:
        console.print(
            f"[yellow]{report.false_positive} false positive(s):[/yellow] the judge approved a "
            "wrong answer. This is the direction that inflates a leaderboard and lets a "
            "regression pass the CI gate."
        )

    if len(per_model) > 1:
        per = Table(title="Agreement by contestant", header_style="bold", title_style="bold")
        per.add_column("contestant", justify="left")
        per.add_column("n", justify="right")
        per.add_column("agreement", justify="right")
        per.add_column("kappa", justify="right")
        for model, model_pairs in sorted(per_model.items()):
            r = compare_verdicts(model_pairs)
            per.add_row(
                model.replace("ollama:", ""),
                str(r.n),
                f"{r.raw_agreement:.1%}",
                f"{r.cohen_kappa:.3f}",
            )
        console.print(per)

    if disagreements:
        console.print("\n[bold]Sample disagreements[/bold] (judge vs arithmetic)")
        for model, item_id, truth_verdict, reason in disagreements[:5]:
            console.print(
                f"  [dim]{model.replace('ollama:', '')} {item_id}[/dim] truth={truth_verdict} "
                f"→ judge said otherwise: {reason[:120]}"
            )

    if json_out:
        payload = {
            "run_id": rid,
            "judge_model": cfg.judge.model,
            "judge_prompt_version": judge_grader.version,
            "overall": report.as_dict(),
            "by_contestant": {
                m: compare_verdicts(p).as_dict() for m, p in sorted(per_model.items())
            },
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        console.print(f"[dim]Wrote {json_out}[/dim]")


def _summaries_for_run(
    cfg: Config, store: Store, rid: str, grader: str | None
) -> list[ModelSummary]:
    """Build a ModelSummary per model, joining generations to judgements if any exist."""
    items = {i.id: i for i in load_golden_set(cfg.run.golden_set, verify_hash=False)}
    judgement_rows = store.judgements(rid)
    judgements = [
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
        for r in judgement_rows
    ]

    out = []
    for model in store.models_in_run(rid):
        gens = [
            generation_from_row(r)
            for r in store.generations(rid, model=model, include_warmup=True)
        ]
        out.append(
            summarize(
                model,
                gens,
                judgements=judgements if judgements else None,
                items=items,
                grader_name=grader,
            )
        )
    return out


@app.command()
def report(
    run_id: Annotated[str, typer.Argument(help="Run id, or 'latest'")] = "latest",
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path(DEFAULT_CONFIG_PATH),
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Also write machine-readable results here")
    ] = None,
    grader: Annotated[
        str | None, typer.Option("--grader", help="Score using only this grader")
    ] = None,
    sort_by: Annotated[
        str, typer.Option("--sort-by", help="accuracy | latency | throughput")
    ] = "accuracy",
) -> None:
    """Aggregate a run into P50/P95 latency, TTFT, throughput and (once judged) accuracy."""
    cfg = load_config(config_path)
    store = Store(cfg.run.db)
    rid = store.latest_run_id() if run_id == "latest" else run_id
    if not rid or not store.get_run(rid):
        console.print(f"[red]No such run:[/red] {run_id}")
        raise typer.Exit(code=1)

    meta = store.get_run(rid)
    assert meta is not None
    summaries = _summaries_for_run(cfg, store, rid, grader)
    if not summaries:
        console.print(f"[yellow]Run {rid} has no generations.[/yellow]")
        raise typer.Exit(code=1)

    keys = {
        "accuracy": lambda s: (-(s.accuracy if s.accuracy is not None else -1)),
        "latency": lambda s: (s.latency_p95_ms or float("inf")),
        "throughput": lambda s: -(s.decode_tps_p50 or 0.0),
    }
    if sort_by not in keys:
        raise typer.BadParameter(f"--sort-by must be one of {sorted(keys)}")
    summaries.sort(key=keys[sort_by])

    judged = any(s.accuracy is not None for s in summaries)

    console.print(
        f"\n[bold]{rid}[/bold]  {meta['status']}  ollama={meta['ollama_version']}  "
        f"prompt={meta['prompt_version']}  golden_set={(meta['golden_set_hash'] or '')[:12]}"
    )

    table = Table(title="Leaderboard", header_style="bold", title_style="bold")
    table.add_column("model", justify="left", no_wrap=True)
    table.add_column("n", justify="right")
    if judged:
        table.add_column("accuracy", justify="right")
        table.add_column("95% CI", justify="right")
    table.add_column("TTFT p50", justify="right")
    table.add_column("TTFT p95", justify="right")
    table.add_column("lat p50", justify="right")
    table.add_column("lat p95", justify="right")
    table.add_column("tok/s p50", justify="right")
    table.add_column("cold", justify="right")
    table.add_column("err", justify="right")
    table.add_column("trunc", justify="right")

    def ms(v: float | None, unit: str = "s") -> str:
        if v is None:
            return "-"
        return f"{v:.0f}ms" if unit == "ms" else f"{v / 1000:.2f}s"

    for s in summaries:
        row = [s.model.replace("ollama:", ""), str(s.n_generations)]
        if judged:
            row.append(f"{s.accuracy:.1%}" if s.accuracy is not None else "-")
            row.append(
                f"{s.accuracy_ci_low:.0%}-{s.accuracy_ci_high:.0%}"
                if s.accuracy_ci_low is not None
                else "-"
            )
        row += [
            ms(s.ttft_p50_ms, "ms"),
            ms(s.ttft_p95_ms, "ms"),
            ms(s.latency_p50_ms),
            ms(s.latency_p95_ms),
            f"{s.decode_tps_p50:.1f}" if s.decode_tps_p50 else "-",
            ms(s.cold_start_ms),
            f"[red]{s.n_errors}[/red]" if s.n_errors else "0",
            f"[yellow]{s.n_truncated}[/yellow]" if s.n_truncated else "0",
        ]
        table.add_row(*row)

    console.print(table)

    n = summaries[0].n_generations
    console.print(
        f"[dim]Percentiles are nearest-rank over n={n} per model: P95 is the "
        f"{n - math.ceil(0.95 * n) + 1}{'st' if n - math.ceil(0.95 * n) + 1 == 1 else 'th'} "
        f"largest sample, so treat it as indicative rather than tight.[/dim]"
    )
    if not judged:
        console.print(
            "[yellow]No judgements yet[/yellow] — accuracy columns appear after "
            "`viveka judge` (Milestone 3)."
        )
    else:
        widest = max(
            (s for s in summaries if s.accuracy_ci_width is not None),
            key=lambda s: s.accuracy_ci_width or 0,
            default=None,
        )
        if widest is not None and widest.accuracy_ci_width:
            console.print(
                f"[dim]Widest accuracy CI spans {widest.accuracy_ci_width:.0%}; models closer "
                f"than that are not statistically separable at this sample size.[/dim]"
            )

    for s in summaries:
        store.save_summary(rid, s.model, s.as_dict())

    if json_out:
        # Per-item verdicts are what let the CI gate name *which* items broke, rather than
        # reporting an aggregate delta that is indistinguishable from noise at n=50 (design §11.3).
        item_verdicts: dict[str, dict[str, str]] = {}
        for judgement in store.judgements(rid):
            if grader and judgement["grader_name"] != grader:
                continue
            item_verdicts.setdefault(judgement["model"], {})[judgement["item_id"]] = judgement[
                "verdict"
            ]

        payload = {
            "run_id": rid,
            "status": meta["status"],
            "golden_set_hash": meta["golden_set_hash"],
            "prompt_version": meta["prompt_version"],
            "viveka_version": meta["viveka_version"],
            "ollama_version": meta["ollama_version"],
            "grader": grader,
            "models": [s.as_dict() for s in summaries],
            "item_verdicts": item_verdicts,
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        console.print(f"[dim]Wrote {json_out} (input for `viveka gate` in Milestone 5).[/dim]")


@app.command()
def gate(
    results: Annotated[
        Path, typer.Option("--results", help="Candidate results JSON from `viveka report --json`")
    ] = Path("results/latest.json"),
    baseline: Annotated[
        Path, typer.Option("--baseline", help="Accepted reference results")
    ] = Path("results/baseline.json"),
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path(DEFAULT_CONFIG_PATH),
    update_baseline: Annotated[
        bool,
        typer.Option("--update-baseline", help="Promote the candidate to be the new baseline"),
    ] = False,
    summary_out: Annotated[
        Path | None,
        typer.Option("--summary", help="Write a markdown summary here (e.g. $GITHUB_STEP_SUMMARY)"),
    ] = None,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write the machine-readable gate report here")
    ] = None,
) -> None:
    """Fail the build if quality regressed. Exits non-zero on any blocking failure.

    Runs in milliseconds on pure JSON, so it works on a hosted CI runner with no GPU and no models
    installed -- the expensive inference happens locally and its evidence is committed
    (design §11.2).
    """
    cfg = load_config(config_path)
    thresholds = thresholds_from_mapping(cfg.gates.model_dump())

    if not results.exists():
        console.print(
            f"[red]No candidate results at {results}.[/red] Produce them with:\n"
            f"  viveka run && viveka judge latest && viveka report latest --json {results}"
        )
        raise typer.Exit(code=1)

    candidate = json.loads(results.read_text(encoding="utf-8"))

    baseline_payload = None
    if baseline.exists():
        baseline_payload = json.loads(baseline.read_text(encoding="utf-8"))
    else:
        console.print(
            f"[yellow]No baseline at {baseline}[/yellow] — checking absolute floors only. "
            "Establish one with --update-baseline once you trust this run."
        )

    # Independent of the baseline: the committed evidence must describe the eval set that is
    # actually on disk now. Catches "edited the golden set, forgot to re-run".
    try:
        current_hash = file_sha256(cfg.run.golden_set)
        if candidate.get("golden_set_hash") != current_hash:
            console.print(
                f"[red]Stale results.[/red] {results} was produced against eval set "
                f"{str(candidate.get('golden_set_hash'))[:12]} but {cfg.run.golden_set} is now "
                f"{current_hash[:12]}. Re-run the benchmark."
            )
            raise typer.Exit(code=1)
    except GoldenSetError:
        console.print("[yellow]Could not hash the local golden set; skipping staleness check.[/yellow]")

    report = evaluate_gate(candidate, baseline_payload, thresholds)

    table = Table(title="Regression gate", header_style="bold", title_style="bold")
    table.add_column("check", justify="left", no_wrap=True)
    table.add_column("model", justify="left")
    table.add_column("", justify="center")
    table.add_column("detail", justify="left")
    for check in report.checks:
        if check.passed:
            mark = "[green]pass[/green]"
        elif check.warning_only:
            mark = "[yellow]warn[/yellow]"
        else:
            mark = "[red]FAIL[/red]"
        table.add_row(
            check.name, (check.model or "").replace("ollama:", ""), mark, check.detail
        )
    console.print(table)

    if report.improvements:
        for model, ids in sorted(report.improvements.items()):
            console.print(
                f"[green]{len(ids)} item(s) improved[/green] on "
                f"{model.replace('ollama:', '')}: {', '.join(ids[:5])}"
            )

    if summary_out:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(
            summarize_markdown(candidate, baseline_payload, report), encoding="utf-8"
        )
        console.print(f"[dim]Wrote markdown summary to {summary_out}[/dim]")

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if update_baseline:
        if not report.passed:
            console.print(
                "[red]Refusing to promote a failing run to baseline.[/red] Fix the failures, or "
                "override deliberately by copying the file by hand."
            )
            raise typer.Exit(code=1)
        baseline.parent.mkdir(parents=True, exist_ok=True)
        baseline.write_text(results.read_text(encoding="utf-8"), encoding="utf-8")
        console.print(f"[green]Baseline updated[/green] from {results} -> {baseline}")

    if report.passed:
        console.print("\n[bold green]GATE PASSED[/bold green]")
        return

    console.print(f"\n[bold red]GATE FAILED[/bold red] — {len(report.failures)} blocking check(s)")
    raise typer.Exit(code=1)


@app.command()
def serve(
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path(DEFAULT_CONFIG_PATH),
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Auto-reload on code changes")] = False,
) -> None:
    """Serve the results API and dashboard (requires the `api` extra)."""
    try:
        import uvicorn
    except ModuleNotFoundError:
        # Escape the brackets: rich would otherwise parse [api] as a markup tag and silently
        # print the instruction without the extra, which is the one detail that matters here.
        console.print(
            r"[red]The API extra is not installed.[/red]  pip install 'viveka\[api]'"
        )
        raise typer.Exit(code=1) from None

    cfg = load_config(config_path)
    # The API reads its database from the environment so a hosted demo can point at a committed
    # read-only copy without editing config.
    os.environ.setdefault("VIVEKA_DB", cfg.run.db)
    os.environ.setdefault("VIVEKA_GOLDEN_SET", cfg.run.golden_set)

    console.print(f"[bold]Viveka API[/bold] on http://{host}:{port}  (db={cfg.run.db})")
    console.print(f"[dim]Docs at http://{host}:{port}/docs[/dim]")
    uvicorn.run("viveka.api.main:app", host=host, port=port, reload=reload)


@app.command()
def config(
    config_path: Annotated[Path, typer.Option("--config", "-c")] = Path(DEFAULT_CONFIG_PATH),
) -> None:
    """Show the resolved, validated configuration."""
    cfg = load_config(config_path)
    console.print_json(data=cfg.snapshot())
    console.print(f"[dim]config_hash={cfg.config_hash()[:16]}…[/dim]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
