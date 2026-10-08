"""LLM-as-judge scoring.

Takes a `Provider`, so the judge is not coupled to Ollama specifically -- but it does rely on
schema-constrained decoding, verified working against `qwen3:8b` in Milestone 0 (design §1.3): a
JSON schema in the request produced parseable JSON on the first attempt, with no prompt begging
and no regex scraping.

Bias mitigations, all deliberate (design §8.2):

* **The judge is never a contestant.** Grading your own output invites self-preference bias;
  `Config.judge_is_a_contestant()` warns if this is violated.
* **Reference-based single-answer grading, not pairwise.** This structurally eliminates position
  bias rather than trying to correct for it afterwards.
* **The judge grades correctness only** -- explicitly not style, length, or formatting. Verbosity
  bias is the other classic judge failure mode.
* **temperature 0 with a fixed seed**, so a re-judge is reproducible and CI noise stays low.

Prompts are versioned and the version is recorded on every judgement: a prompt edit is exactly
what the CI gate exists to catch, so an unversioned judgement cannot be compared across runs.
"""

from __future__ import annotations

import json
import time
from typing import Any

from viveka.core.providers.base import Provider
from viveka.core.types import GoldenItem, Score

__all__ = ["JUDGE_PROMPT_VERSION", "JUDGE_SCHEMA", "LLMJudge", "build_judge_prompt"]

JUDGE_PROMPT_VERSION = "v1"

#: Constrains decoding so the response is guaranteed-shaped JSON.
JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "incorrect"]},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "confidence", "reasoning"],
}

_SYSTEM = (
    "You are a strict grading assistant. You judge only whether a candidate answer is factually "
    "correct with respect to the reference. You never reward verbosity, confidence, or writing "
    "style. Respond with JSON only."
)

_TEMPLATE = """Grade the candidate answer against the reference.

Question:
{question}

Reference answer:
{reference}
{extra}
Candidate answer:
{candidate}

Judge ONLY whether the candidate's final answer matches the reference in substance. Ignore
differences in wording, formatting, length, and explanation quality. A candidate that reaches the
right conclusion by a different route is correct. A candidate that hedges without committing to
the reference's conclusion is incorrect.

Respond with JSON: {{"verdict": "correct" | "incorrect", "confidence": 0.0-1.0, "reasoning": "one sentence"}}"""


def build_judge_prompt(item: GoldenItem, candidate: str) -> str:
    """Render the judge prompt, including any enumerated acceptable answers.

    TruthfulQA ships `correct_answers` / `incorrect_answers` lists, which are far better judge
    context than a single `best_answer` string -- they tell the judge what the acceptable answer
    *space* looks like instead of forcing a near-string-match against one phrasing.
    """
    parts = []
    if rationale := item.reference_rationale:
        parts.append(f"\nReference reasoning:\n{rationale}\n")
    correct = item.metadata.get("correct_answers") or []
    incorrect = item.metadata.get("incorrect_answers") or []
    if correct:
        parts.append("\nAlso acceptable:\n" + "\n".join(f"- {a}" for a in correct[:8]) + "\n")
    if incorrect:
        parts.append(
            "\nExplicitly NOT acceptable:\n" + "\n".join(f"- {a}" for a in incorrect[:8]) + "\n"
        )

    # An empty response is a real case (a failed generation); make it visible to the judge
    # rather than sending a blank section it might charitably fill in.
    candidate_text = candidate.strip() or "(the model produced no output)"
    return _TEMPLATE.format(
        question=item.question,
        reference=item.reference_answer,
        extra="".join(parts),
        candidate=candidate_text,
    )


class LLMJudge:
    """Grades a response with a language model, returning a structured verdict."""

    name = "llm_judge"

    def __init__(self, provider: Provider, *, version: str = JUDGE_PROMPT_VERSION) -> None:
        self.provider = provider
        self.version = version

    @property
    def judge_model(self) -> str:
        return self.provider.name

    def grade(self, item: GoldenItem, text: str, *, truncated: bool = False) -> Score:
        """Score one response. Never raises; failures come back as `verdict="error"`.

        `truncated` is accepted for protocol compatibility but not used: the judge reads the whole
        response semantically rather than relying on where text happens to stop, so a cut-off
        answer is something it can legitimately judge as incorrect.
        """
        prompt = build_judge_prompt(item, text)
        started = time.perf_counter()
        completion = self.provider.complete(prompt, system=_SYSTEM)
        latency_ms = (time.perf_counter() - started) * 1000.0

        if not completion.ok:
            return Score(
                verdict="error",
                score=0.0,
                latency_ms=latency_ms,
                error=f"judge call failed: {completion.error}",
            )

        try:
            parsed = json.loads(completion.text)
        except json.JSONDecodeError as exc:
            # Should not happen with schema-constrained decoding, but a judge that silently
            # scores 0 on a parse failure would quietly corrupt the leaderboard.
            return Score(
                verdict="error",
                score=0.0,
                latency_ms=latency_ms,
                error=f"judge returned unparseable JSON: {exc}",
                raw={"text": completion.text[:2000]},
            )

        verdict = str(parsed.get("verdict", "")).strip().lower()
        if verdict not in {"correct", "incorrect"}:
            return Score(
                verdict="error",
                score=0.0,
                latency_ms=latency_ms,
                error=f"judge returned unknown verdict {verdict!r}",
                raw=parsed,
            )

        confidence = parsed.get("confidence")
        try:
            confidence = min(1.0, max(0.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = None

        return Score(
            verdict="correct" if verdict == "correct" else "incorrect",
            score=1.0 if verdict == "correct" else 0.0,
            confidence=confidence,
            reasoning=str(parsed.get("reasoning") or "").strip() or None,
            latency_ms=latency_ms,
            raw=parsed,
        )
