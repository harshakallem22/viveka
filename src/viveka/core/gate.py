"""CI regression gating: decide whether a result set is allowed to merge.

Stdlib only.

The obstacle this design works around (design §11.1): **a free GitHub Actions runner cannot run
this eval.** No GPU, ~14GB disk, and CPU-only 8B inference would take minutes per item. Pretending
otherwise produces a workflow that is permanently red or quietly disabled.

So evidence generation and gate enforcement are separated. The expensive inference runs locally on
the developer's machine and its output is committed as `results/latest.json`; CI runs *this* logic,
which is pure arithmetic over two JSON files and finishes in milliseconds. The gate is real -- it
fails builds on real numbers -- while the GPU work happens where a GPU exists.

That design has one obvious hole: committed evidence can be stale or hand-edited. Three defences:

* `golden_set_hash` must match between baseline and candidate, so deleting the three hardest items
  to turn a red build green is caught (an eval-set change makes the comparison meaningless, not
  merely suspicious).
* `prompt_version` mismatch is surfaced, because a prompt edit invalidates comparison.
* The workflow separately asserts the results file is newer than the code that produced it (git
  timestamps, done in CI where git is available rather than here).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "GateCheck",
    "GateReport",
    "GateThresholds",
    "evaluate_gate",
]


@dataclass(frozen=True)
class GateThresholds:
    """Mirrors the `gates:` block of viveka.yaml."""

    min_accuracy: float = 0.55
    max_p95_latency_ms: float = 45000.0
    max_accuracy_drop: float = 0.10
    max_regressed_items: int = 3
    require_same_golden_set: bool = True


@dataclass(frozen=True)
class GateCheck:
    """One pass/fail assertion, with enough detail to act on without rerunning anything."""

    name: str
    passed: bool
    detail: str
    model: str | None = None
    #: Warnings surface a problem without failing the build.
    warning_only: bool = False

    @property
    def blocking(self) -> bool:
        return not self.passed and not self.warning_only


@dataclass(frozen=True)
class GateReport:
    checks: list[GateCheck] = field(default_factory=list)
    #: Item ids that flipped correct -> incorrect, per model.
    regressions: dict[str, list[str]] = field(default_factory=dict)
    #: Item ids that flipped incorrect -> correct, per model.
    improvements: dict[str, list[str]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not any(c.blocking for c in self.checks)

    @property
    def failures(self) -> list[GateCheck]:
        return [c for c in self.checks if c.blocking]

    @property
    def warnings(self) -> list[GateCheck]:
        return [c for c in self.checks if not c.passed and c.warning_only]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [asdict(c) for c in self.checks],
            "regressions": self.regressions,
            "improvements": self.improvements,
        }


def _models_by_name(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {m["model"]: m for m in payload.get("models", [])}


def _item_verdicts(payload: Mapping[str, Any], model: str) -> dict[str, str]:
    """Per-item verdicts, if the report carried them."""
    items = payload.get("item_verdicts") or {}
    got = items.get(model) or {}
    return {str(k): str(v) for k, v in got.items()}


def evaluate_gate(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    thresholds: GateThresholds,
) -> GateReport:
    """Compare a candidate result set against absolute floors and (if given) a baseline.

    Returns a report rather than raising, so the caller can print every problem at once instead of
    forcing a fix-one-rerun loop.
    """
    checks: list[GateCheck] = []
    regressions: dict[str, list[str]] = {}
    improvements: dict[str, list[str]] = {}

    cand_models = _models_by_name(candidate)
    if not cand_models:
        checks.append(
            GateCheck("results_present", False, "candidate results contain no models")
        )
        return GateReport(checks=checks)

    # --- integrity of the comparison itself -------------------------------------------------
    if baseline is not None:
        cand_hash = candidate.get("golden_set_hash")
        base_hash = baseline.get("golden_set_hash")
        same_set = bool(cand_hash) and cand_hash == base_hash
        if thresholds.require_same_golden_set:
            checks.append(
                GateCheck(
                    "golden_set_unchanged",
                    same_set,
                    f"candidate={str(cand_hash)[:12]} baseline={str(base_hash)[:12]}"
                    + (
                        ""
                        if same_set
                        else " — the eval set changed, so accuracy deltas are not comparable. "
                        "Re-run the benchmark and update the baseline deliberately."
                    ),
                )
            )
        if candidate.get("prompt_version") != baseline.get("prompt_version"):
            checks.append(
                GateCheck(
                    "prompt_version_unchanged",
                    False,
                    f"prompt version changed "
                    f"{baseline.get('prompt_version')} -> {candidate.get('prompt_version')}; "
                    "accuracy differences may be caused by the prompt, not the code",
                    warning_only=True,
                )
            )

        missing = sorted(set(_models_by_name(baseline)) - set(cand_models))
        if missing:
            checks.append(
                GateCheck(
                    "no_models_dropped",
                    False,
                    f"models present in baseline but missing from candidate: {missing}. "
                    "Removing a failing model is not the same as fixing it.",
                )
            )

    # --- per-model thresholds ---------------------------------------------------------------
    base_models = _models_by_name(baseline) if baseline is not None else {}

    for name, cand in sorted(cand_models.items()):
        accuracy = cand.get("accuracy")
        p95 = cand.get("latency_p95_ms")

        if accuracy is None:
            checks.append(
                GateCheck(
                    "accuracy_measured",
                    False,
                    "no accuracy recorded — run `viveka judge` before gating",
                    model=name,
                    warning_only=True,
                )
            )
        else:
            ok = accuracy >= thresholds.min_accuracy
            checks.append(
                GateCheck(
                    "min_accuracy",
                    ok,
                    f"{accuracy:.1%} vs floor {thresholds.min_accuracy:.1%}",
                    model=name,
                )
            )

        if p95 is not None:
            ok = p95 <= thresholds.max_p95_latency_ms
            checks.append(
                GateCheck(
                    "max_p95_latency",
                    ok,
                    f"{p95 / 1000:.2f}s vs ceiling {thresholds.max_p95_latency_ms / 1000:.2f}s",
                    model=name,
                )
            )

        base = base_models.get(name)
        if base is None:
            continue

        base_accuracy = base.get("accuracy")
        if accuracy is not None and base_accuracy is not None:
            drop = base_accuracy - accuracy
            ok = drop <= thresholds.max_accuracy_drop
            direction = "drop" if drop > 0 else "gain"
            checks.append(
                GateCheck(
                    "max_accuracy_drop",
                    ok,
                    f"{base_accuracy:.1%} -> {accuracy:.1%} ({direction} {abs(drop):.1%}), "
                    f"allowed drop {thresholds.max_accuracy_drop:.1%}",
                    model=name,
                )
            )

        # Item-level comparison. More sensitive and far more actionable than an aggregate delta:
        # it names the items that broke. Enabled by the stable per-item ids in the golden set.
        cand_items = _item_verdicts(candidate, name)
        base_items = _item_verdicts(baseline or {}, name)
        if cand_items and base_items:
            broke = sorted(
                item_id
                for item_id, verdict in cand_items.items()
                if base_items.get(item_id) == "correct" and verdict == "incorrect"
            )
            fixed = sorted(
                item_id
                for item_id, verdict in cand_items.items()
                if base_items.get(item_id) == "incorrect" and verdict == "correct"
            )
            if broke:
                regressions[name] = broke
            if fixed:
                improvements[name] = fixed

            ok = len(broke) <= thresholds.max_regressed_items
            checks.append(
                GateCheck(
                    "max_regressed_items",
                    ok,
                    f"{len(broke)} item(s) went correct->incorrect "
                    f"(limit {thresholds.max_regressed_items})"
                    + (f": {', '.join(broke[:5])}" if broke else "")
                    + (f"; {len(fixed)} improved" if fixed else ""),
                    model=name,
                )
            )

    return GateReport(checks=checks, regressions=regressions, improvements=improvements)


def summarize_markdown(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    report: GateReport,
) -> str:
    """Render a markdown table for a PR comment / GitHub step summary."""
    lines: list[str] = []
    verdict = "PASSED" if report.passed else "FAILED"
    lines.append(f"## Viveka regression gate: **{verdict}**")
    lines.append("")
    lines.append(f"Run `{candidate.get('run_id', '?')}` · eval set `{str(candidate.get('golden_set_hash'))[:12]}` · prompt `{candidate.get('prompt_version')}`")
    lines.append("")

    base_models = _models_by_name(baseline) if baseline else {}
    lines.append("| model | accuracy | Δ vs baseline | P95 latency | Δ | tok/s |")
    lines.append("|---|---|---|---|---|---|")
    for name, m in sorted(_models_by_name(candidate).items()):
        b = base_models.get(name, {})
        acc = m.get("accuracy")
        bacc = b.get("accuracy")
        p95 = m.get("latency_p95_ms")
        bp95 = b.get("latency_p95_ms")
        acc_s = f"{acc:.1%}" if acc is not None else "-"
        dacc = f"{(acc - bacc) * 100:+.1f}pp" if acc is not None and bacc is not None else "-"
        p95_s = f"{p95 / 1000:.2f}s" if p95 is not None else "-"
        dp95 = f"{(p95 - bp95) / 1000:+.2f}s" if p95 is not None and bp95 is not None else "-"
        tps = m.get("decode_tps_p50")
        lines.append(
            f"| `{name}` | {acc_s} | {dacc} | {p95_s} | {dp95} | "
            f"{f'{tps:.1f}' if tps else '-'} |"
        )

    if report.failures:
        lines += ["", "### Blocking failures", ""]
        lines += [
            f"- **{c.name}**{f' (`{c.model}`)' if c.model else ''}: {c.detail}"
            for c in report.failures
        ]
    if report.warnings:
        lines += ["", "### Warnings", ""]
        lines += [
            f"- {c.name}{f' (`{c.model}`)' if c.model else ''}: {c.detail}"
            for c in report.warnings
        ]
    if report.regressions:
        lines += ["", "### Items that regressed", ""]
        for model, ids in sorted(report.regressions.items()):
            lines.append(f"- `{model}`: {', '.join(f'`{i}`' for i in ids)}")

    return "\n".join(lines) + "\n"


def thresholds_from_mapping(raw: Mapping[str, Any] | None) -> GateThresholds:
    """Build thresholds from a config mapping, ignoring unrelated keys."""
    if not raw:
        return GateThresholds()
    fields: Sequence[str] = (
        "min_accuracy",
        "max_p95_latency_ms",
        "max_accuracy_drop",
        "max_regressed_items",
        "require_same_golden_set",
    )
    return GateThresholds(**{f: raw[f] for f in fields if f in raw})
