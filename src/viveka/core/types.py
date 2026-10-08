"""Core data types for the Viveka eval harness.

Design invariant: this module imports **nothing outside
the standard library**. It is the vocabulary every other layer speaks, so it must stay
trivially importable and testable.

Every type here is frozen. Eval results are historical facts -- once a generation has been
recorded it must not be mutable, or the audit trail the CI gate depends on is worthless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "Completion",
    "Generation",
    "GoldenItem",
    "Judgement",
    "Score",
    "Timing",
    "Verdict",
]

# "error" is deliberately distinct from "incorrect": a grader that could not parse an answer
# is a *harness* failure, and collapsing it into "incorrect" would make a broken regex look
# like a bad model. See design §8.1.
Verdict = Literal["correct", "incorrect", "error"]

_NS_PER_MS = 1_000_000.0


def _ns_to_ms(value: Any) -> float | None:
    """Ollama reports durations in nanoseconds; we store milliseconds."""
    if value is None:
        return None
    return float(value) / _NS_PER_MS


@dataclass(frozen=True)
class GoldenItem:
    """One frozen eval case.

    `grader` selects which grader scores this item, so the runner needs no per-source
    branching -- adding a new item type is a data change, not a code change (constraint C5).
    """

    id: str
    source: str
    category: str
    question: str
    reference_answer: str
    grader: str
    reference_rationale: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize in a stable key order so the JSONL diffs cleanly."""
        out: dict[str, Any] = {
            "id": self.id,
            "source": self.source,
            "category": self.category,
            "question": self.question,
            "reference_answer": self.reference_answer,
        }
        if self.reference_rationale is not None:
            out["reference_rationale"] = self.reference_rationale
        out["grader"] = self.grader
        out["metadata"] = self.metadata
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GoldenItem:
        return cls(
            id=raw["id"],
            source=raw["source"],
            category=raw["category"],
            question=raw["question"],
            reference_answer=raw["reference_answer"],
            grader=raw["grader"],
            reference_rationale=raw.get("reference_rationale"),
            metadata=raw.get("metadata") or {},
        )


@dataclass(frozen=True)
class Timing:
    """Timing for a single generation.

    Two clocks are mixed on purpose (design §6.2-6.3):

    * `total_ms` / `ttft_ms` are **client-side** wall clock. TTFT is defined by what a user
      perceives, so it must include HTTP and queueing.
    * `load_ms` / `prompt_eval_ms` / `decode_ms` are **server-reported**. Throughput derived
      from `decode_ms` excludes weight loading, which is the only way to get a number
      comparable across models -- measured on this machine, a cold call's wall-clock
      throughput understated true decode speed by 89x.
    """

    total_ms: float
    ttft_ms: float | None = None
    load_ms: float | None = None
    prompt_eval_ms: float | None = None
    decode_ms: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def decode_tps(self) -> float | None:
        """Decode-only throughput. The headline, model-comparable number."""
        if not self.output_tokens or not self.decode_ms:
            return None
        return self.output_tokens / (self.decode_ms / 1000.0)

    @property
    def end_to_end_tps(self) -> float | None:
        """Wall-clock throughput. Degraded by model load and prompt eval; what a user feels."""
        if not self.output_tokens or not self.total_ms:
            return None
        return self.output_tokens / (self.total_ms / 1000.0)

    @classmethod
    def from_ollama(
        cls, payload: dict[str, Any], *, total_ms: float, ttft_ms: float | None
    ) -> Timing:
        """Build from the final chunk of an Ollama stream.

        Pure dict arithmetic, so it is unit-testable against a recorded payload without a
        live server or an HTTP client.
        """
        return cls(
            total_ms=total_ms,
            ttft_ms=ttft_ms,
            load_ms=_ns_to_ms(payload.get("load_duration")),
            prompt_eval_ms=_ns_to_ms(payload.get("prompt_eval_duration")),
            decode_ms=_ns_to_ms(payload.get("eval_duration")),
            prompt_tokens=payload.get("prompt_eval_count"),
            output_tokens=payload.get("eval_count"),
        )


@dataclass(frozen=True)
class Completion:
    """What a Provider returns: timed text, with no eval-harness identity attached.

    Keeping identity out of this type is what lets a Provider be anything that turns a
    prompt into timed text -- an Ollama model now, a RAG pipeline or a fine-tuned adapter
    in later projects (design §5.2).

    `error` is captured rather than raised so one bad item cannot abort a 30-minute run.
    """

    text: str
    timing: Timing
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class Generation:
    """A Completion bound to a run, item and model -- the unit we persist."""

    run_id: str
    item_id: str
    model: str
    repeat_idx: int
    completion: Completion
    is_warmup: bool = False
    #: Set once persisted; the join key for Judgement.
    id: int | None = None


@dataclass(frozen=True)
class Score:
    """What a Grader returns: a verdict with no eval-harness identity attached.

    Mirrors the Completion -> Generation split (a grader is handed text and a reference; it has
    no idea which run or generation row it is scoring). The caller attaches identity via
    `to_judgement`, which keeps graders trivially unit-testable.
    """

    verdict: Verdict
    score: float
    confidence: float | None = None
    reasoning: str | None = None
    latency_ms: float | None = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_judgement(
        self,
        generation_id: int,
        *,
        grader_name: str,
        grader_version: str = "v1",
        judge_model: str | None = None,
    ) -> Judgement:
        return Judgement(
            generation_id=generation_id,
            grader_name=grader_name,
            grader_version=grader_version,
            judge_model=judge_model,
            verdict=self.verdict,
            score=self.score,
            confidence=self.confidence,
            reasoning=self.reasoning,
            latency_ms=self.latency_ms,
            error=self.error,
            raw=self.raw,
        )


@dataclass(frozen=True)
class Judgement:
    """One grader's scoring of one Generation.

    `score` is a float rather than a bool so a graded rubric (e.g. a 0-5 faithfulness scale
    in the RAG project) fits without a schema migration.
    """

    generation_id: int
    grader_name: str
    verdict: Verdict
    score: float
    grader_version: str = "v1"
    judge_model: str | None = None  # None for deterministic graders
    confidence: float | None = None
    reasoning: str | None = None
    latency_ms: float | None = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_scorable(self) -> bool:
        """False for harness failures, which must be excluded from accuracy, not counted wrong."""
        return self.verdict != "error"
