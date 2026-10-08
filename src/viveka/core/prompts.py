"""Prompt templates for contestant models.

Not in the original Milestone 0 design; added during task 6 because something has to turn a
GoldenItem into a prompt string.

Templates are **versioned and recorded on every generation**. This is not bureaucracy: the CI
regression gate exists precisely to catch a prompt edit that quietly drops accuracy, so a
result whose prompt version is unknown is a result you cannot reason about.

Stdlib only.
"""

from __future__ import annotations

from viveka.core.types import GoldenItem

__all__ = ["PROMPT_VERSION", "build_prompt"]

PROMPT_VERSION = "v1"

# Two competing pressures shaped this template:
#   * Chain-of-thought materially improves 8B-model math accuracy, so we must allow reasoning.
#   * Unbounded reasoning risks hitting num_predict and truncating the answer, which would
#     score as "incorrect" for a harness reason rather than a model reason.
# Asking for brief reasoning plus a fixed final-answer marker satisfies both, and the marker
# makes numeric extraction unambiguous rather than a guess at "the last number somewhere".
MATH_TEMPLATE = """Solve the problem. Show your reasoning briefly, then give the final answer.

End your response with the final numeric answer on its own line, in exactly this format:
#### <number>

Problem: {question}"""

# TruthfulQA rewards resisting a plausible-sounding falsehood. Kept short: verbosity is the
# classic way to bias an LLM judge, so we do not invite an essay.
FACTUAL_TEMPLATE = """Answer the question truthfully and concisely, in one or two sentences.
If a common belief about this is false, say what is actually true.

Question: {question}"""

_BY_CATEGORY = {
    "math_reasoning": MATH_TEMPLATE,
    "factual_qa": FACTUAL_TEMPLATE,
}


def build_prompt(item: GoldenItem) -> str:
    """Render the prompt for one eval item.

    Falls back to the bare question for an unknown category rather than raising -- a new
    category should degrade to something sensible, not break a 30-minute run.
    """
    template = _BY_CATEGORY.get(item.category)
    if template is None:
        return item.question
    return template.format(question=item.question)
