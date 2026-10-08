"""Run orchestration: golden set x models -> persisted generations.

Stdlib only apart from the types it moves around.

The execution order is the load-bearing part. Generations are **batched by model**: every item
for model A, then unload, then every item for model B. Measured on this machine, a model load
costs 5-7s and 16GB cannot hold two 8B models comfortably, so interleaving models (or
interleaving generation with judging) would pay that reload on nearly every call and turn a
20-minute run into hours. See design §1.4.

The runner emits progress events rather than printing. Rendering belongs to the CLI, which
keeps `core` free of presentation concerns and lets the API reuse this code untouched.
"""

from __future__ import annotations

import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from viveka.core.prompts import PROMPT_VERSION, build_prompt
from viveka.core.providers.base import Provider
from viveka.core.store import Store, new_run_id
from viveka.core.types import Completion, Generation, GoldenItem

__all__ = ["RunConfig", "RunProgress", "Runner", "host_info"]


def host_info() -> dict[str, Any]:
    """Machine provenance. Latency numbers are meaningless without knowing the hardware."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


@dataclass(frozen=True)
class RunConfig:
    """Execution knobs. Mirrors the `run:` block of viveka.yaml."""

    repeats: int = 1
    warmup: bool = True
    resume: bool = True
    #: Sampling options forwarded to the provider (temperature, seed, num_predict).
    defaults: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunProgress:
    """One event emitted as work completes."""

    model: str
    model_index: int
    model_total: int
    item_index: int
    item_total: int
    item_id: str
    completion: Completion | None
    is_warmup: bool = False
    skipped: bool = False

    @property
    def failed(self) -> bool:
        return self.completion is not None and not self.completion.ok


ProgressHook = Callable[[RunProgress], None]


class Runner:
    """Executes a benchmark run and persists every generation."""

    def __init__(
        self,
        store: Store,
        *,
        config: RunConfig | None = None,
        on_progress: ProgressHook | None = None,
    ) -> None:
        self.store = store
        self.config = config or RunConfig()
        self.on_progress = on_progress

    def _emit(self, event: RunProgress) -> None:
        if self.on_progress is not None:
            self.on_progress(event)

    def run(
        self,
        items: Sequence[GoldenItem],
        providers: Sequence[Provider],
        *,
        run_id: str | None = None,
        golden_set_hash: str | None = None,
        config_snapshot: dict[str, Any] | None = None,
        ollama_version: str | None = None,
    ) -> str:
        """Generate outputs for every (model, item, repeat) and return the run id.

        Individual failures are recorded and stepped over -- a timeout on item 12 must not
        discard the 11 results already paid for.
        """
        from viveka import __version__

        rid = run_id or new_run_id()
        self.store.create_run(
            rid,
            config=config_snapshot or {},
            golden_set_hash=golden_set_hash,
            prompt_version=PROMPT_VERSION,
            viveka_version=__version__,
            ollama_version=ollama_version,
            host=host_info(),
        )

        repeats = max(1, self.config.repeats)
        defaults = dict(self.config.defaults or {})
        total_models = len(providers)

        try:
            for m_idx, provider in enumerate(providers, start=1):
                self._run_one_model(
                    provider,
                    items,
                    rid,
                    repeats=repeats,
                    defaults=defaults,
                    model_index=m_idx,
                    model_total=total_models,
                )
        except BaseException:
            # Includes KeyboardInterrupt: mark the run interrupted, but keep every row already
            # written. Resuming with the same run_id picks up where this left off.
            self.store.finish_run(rid, status="failed")
            raise

        self.store.finish_run(rid, status="complete")
        return rid

    def _run_one_model(
        self,
        provider: Provider,
        items: Sequence[GoldenItem],
        run_id: str,
        *,
        repeats: int,
        defaults: dict[str, Any],
        model_index: int,
        model_total: int,
    ) -> None:
        model = provider.name
        already = (
            self.store.completed_keys(run_id, model=model) if self.config.resume else set()
        )
        unit_total = len(items) * repeats

        # Warmup first: the cold-start cost must not land in the measured distribution.
        # It is still *recorded* (is_warmup=1) because "first-token cost after eviction" is
        # itself a publishable number, and excluding data is worse than flagging it (§6.1).
        if self.config.warmup:
            warm = provider.warmup()
            self.store.record_generation(
                Generation(
                    run_id=run_id,
                    item_id="__warmup__",
                    model=model,
                    repeat_idx=0,
                    completion=warm,
                    is_warmup=True,
                ),
                prompt_version=PROMPT_VERSION,
            )
            self._emit(
                RunProgress(
                    model=model,
                    model_index=model_index,
                    model_total=model_total,
                    item_index=0,
                    item_total=unit_total,
                    item_id="__warmup__",
                    completion=warm,
                    is_warmup=True,
                )
            )

        unit = 0
        for repeat_idx in range(repeats):
            for item in items:
                unit += 1
                if (item.id, repeat_idx) in already:
                    self._emit(
                        RunProgress(
                            model=model,
                            model_index=model_index,
                            model_total=model_total,
                            item_index=unit,
                            item_total=unit_total,
                            item_id=item.id,
                            completion=None,
                            skipped=True,
                        )
                    )
                    continue

                completion = provider.complete(build_prompt(item), **defaults)
                self.store.record_generation(
                    Generation(
                        run_id=run_id,
                        item_id=item.id,
                        model=model,
                        repeat_idx=repeat_idx,
                        completion=completion,
                    ),
                    prompt_version=PROMPT_VERSION,
                )
                self._emit(
                    RunProgress(
                        model=model,
                        model_index=model_index,
                        model_total=model_total,
                        item_index=unit,
                        item_total=unit_total,
                        item_id=item.id,
                        completion=completion,
                    )
                )

        # Free the weights before the next model loads, or both compete for 16GB.
        provider.unload()
