"""Viveka -- an LLM evaluation harness.

LLM-as-judge correctness scoring, SRE-style latency metrics (P50/P95, TTFT), throughput,
and CI regression gating.

This module is a curated facade. `viveka.core` remains the real home of the library, so
internals stay free to move while downstream projects get a stable import path::

    from viveka import load_golden_set, Timing

The public surface is deliberately small; `viveka.core` holds the real implementations.
"""

from viveka.core import (
    Completion,
    Generation,
    GoldenItem,
    GoldenSetError,
    Judgement,
    Timing,
    Verdict,
    load_golden_set,
)

__version__ = "0.1.0"

__all__ = [
    "Completion",
    "Generation",
    "GoldenItem",
    "GoldenSetError",
    "Judgement",
    "Timing",
    "Verdict",
    "__version__",
    "load_golden_set",
]
