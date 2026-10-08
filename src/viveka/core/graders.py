"""Deterministic graders and the grader registry.

Stdlib only.

`NumericExactMatch` is the load-bearing grader for GSM8K, and it matters for two reasons beyond
scoring: it is free and deterministic, and because GSM8K has objective ground truth it doubles as
the **reference against which the LLM judge is validated** (design §8.3). If this grader is wrong,
the judge reliability number is meaningless too.

Every extraction rule below traces to an observed failure, not a hypothetical:

* `34.00` vs `34` mis-scored 10% of a real smoke sample -> compare numerically, never as strings.
* `1,430` appears in ~1% of GSM8K answers -> strip thousands separators.
* A response with no extractable number returns `error`, not `incorrect` -> a broken extraction
  rule must never masquerade as a model that got the answer wrong.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from viveka.core.types import GoldenItem, Score

__all__ = [
    "GRADER_VERSION",
    "Grader",
    "NumericExactMatch",
    "extract_final_number",
    "get_grader",
    "parse_number",
]

GRADER_VERSION = "v1"

#: The marker our prompt asks for. Last match wins -- a model may restate it.
_MARKER_RE = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")
#: LaTeX \boxed{42}, which reasoning-styled models emit unprompted.
_BOXED_RE = re.compile(r"\\boxed\s*\{\s*\$?\s*(-?[\d,]*\.?\d+)\s*\}")
#: Any number. '$' and '%' are excluded from the capture so they never reach float().
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_number(raw: str) -> float | None:
    """Normalize an extracted numeric string to a float, or None if it is not a number."""
    cleaned = raw.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    if not cleaned or cleaned in {"-", "."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_final_number(text: str, *, allow_loose: bool = True) -> float | None:
    """Pull the model's final numeric answer out of a free-text response.

    Tried in order of decreasing confidence:

    1. The last ``#### N`` marker -- our prompt asks for it, and compliance was 90-100% across
       the three benchmarked models, so it is the reliable path.
    2. The last ``\\boxed{N}``.
    3. (loose) The last number on the last line that contains one.

    Rule 3 scans from the end because reasoning text is full of intermediate arithmetic; the
    conclusion is at the bottom. It recovered 7 of 150 real responses (4.7%) that rules 1-2
    missed, so it earns its place -- but it is a heuristic, and `allow_loose=False` disables it
    for responses where "the last line" is not a conclusion (see `NumericExactMatch.grade`).

    Returns None when nothing usable is present, which the caller must surface as `error` rather
    than `incorrect`.
    """
    if not text:
        return None

    for pattern in (_MARKER_RE, _BOXED_RE):
        matches = pattern.findall(text)
        if matches:
            value = parse_number(matches[-1])
            if value is not None:
                return value

    if not allow_loose:
        return None

    for line in reversed(text.strip().splitlines()):
        numbers = _NUMBER_RE.findall(line)
        if numbers:
            value = parse_number(numbers[-1])
            if value is not None:
                return value
    return None


@runtime_checkable
class Grader(Protocol):
    """Scores one model response against one golden item."""

    name: str
    version: str

    def grade(self, item: GoldenItem, text: str, *, truncated: bool = False) -> Score:
        """Score one response.

        Args:
            truncated: the response hit the token cap, so it may stop mid-sentence. Graders that
                use positional heuristics ("the last line holds the answer") must not trust them
                here. Ignored by graders that do not care.

        Must not raise: a grader failure is reported as `Score(verdict="error")`.
        """
        ...


class NumericExactMatch:
    """Compares the model's final number against the reference answer.

    Deterministic and free, so it is preferred wherever an objective answer exists.
    """

    name = "numeric_exact_match"
    version = GRADER_VERSION

    def __init__(self, *, rel_tol: float = 1e-9, abs_tol: float = 1e-6) -> None:
        # Tolerance rather than == so 34.00 and 34 agree, and so float parsing of large values
        # cannot fail on a representation detail.
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def grade(self, item: GoldenItem, text: str, *, truncated: bool = False) -> Score:
        reference = parse_number(item.reference_answer)
        if reference is None:
            # The golden set is validated at build time, so this means the eval data is broken,
            # not the model. Never blame the model for it.
            return Score(
                verdict="error",
                score=0.0,
                error=f"reference answer {item.reference_answer!r} is not numeric",
            )

        # On a truncated response the "last line" is mid-sentence, so the loose heuristic is
        # meaningless there. Observed in the wild: a cut-off response ending "4. Their mother then
        # made 4 pies" yields 4 from a *list marker*. Here that happened to be wrong anyway, but
        # had the reference been 4 it would have scored a false correct and inflated the
        # leaderboard. A model that never emitted an answer is `incorrect`, not accidentally right.
        predicted = extract_final_number(text, allow_loose=not truncated)
        if predicted is None:
            if truncated:
                return Score(
                    verdict="incorrect",
                    score=0.0,
                    reasoning=(
                        "response hit the token cap without producing a final answer; refusing "
                        "to guess a number from a mid-sentence cutoff"
                    ),
                    raw={"reference": reference, "truncated": True},
                )
            return Score(
                verdict="error",
                score=0.0,
                reasoning="no numeric answer could be extracted from the response",
                error="unextractable",
                raw={"reference": reference},
            )

        correct = math.isclose(predicted, reference, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        return Score(
            verdict="correct" if correct else "incorrect",
            score=1.0 if correct else 0.0,
            reasoning=f"extracted {predicted:g}, reference {reference:g}",
            raw={"predicted": predicted, "reference": reference},
        )


#: Populated with deterministic graders. The LLM judge is registered by the caller, since it
#: needs a Provider and `viveka.core.graders` must stay free of network concerns.
GRADERS: dict[str, Grader] = {
    NumericExactMatch.name: NumericExactMatch(),
}


def get_grader(name: str, registry: Mapping[str, Grader] | None = None) -> Grader:
    table = registry if registry is not None else GRADERS
    try:
        return table[name]
    except KeyError:
        raise KeyError(
            f"unknown grader {name!r}; registered: {sorted(table)}. "
            "Items select their grader via the golden set's `grader` field."
        ) from None
