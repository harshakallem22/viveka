"""viveka.core -- the reusable eval library.

This subpackage is the part downstream projects import. It must never depend on the API or
dashboard layers (constraint C1), which is enforced by
tests/test_import_boundary.py rather than left to convention.
"""

from viveka.core.datasets import GoldenSetError, file_sha256, load_golden_set, load_meta
from viveka.core.metrics import ModelSummary, percentile, summarize, wilson_interval
from viveka.core.prompts import PROMPT_VERSION, build_prompt
from viveka.core.providers import Provider
from viveka.core.runner import RunConfig, Runner
from viveka.core.store import Store, new_run_id
from viveka.core.types import (
    Completion,
    Generation,
    GoldenItem,
    Judgement,
    Timing,
    Verdict,
)

__all__ = [
    "PROMPT_VERSION",
    "Completion",
    "Generation",
    "GoldenItem",
    "GoldenSetError",
    "Judgement",
    "ModelSummary",
    "Provider",
    "RunConfig",
    "Runner",
    "Store",
    "Timing",
    "Verdict",
    "build_prompt",
    "file_sha256",
    "load_golden_set",
    "load_meta",
    "new_run_id",
    "percentile",
    "summarize",
    "wilson_interval",
]
