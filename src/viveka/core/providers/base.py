"""The Provider seam.

This is the most important extension point in the library (design §5.2). A Provider is
anything that turns a prompt into *timed text*. Ollama models are one implementation; the RAG
pipeline in Project 2 and the fine-tuned adapter in Project 4 are others.

Keeping eval-harness identity (run/item/model) out of `Completion` is what makes that
substitution possible -- the Runner attaches identity, so a Provider never needs to know it is
part of an eval at all.

Stdlib only.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from viveka.core.types import Completion

__all__ = ["Provider", "ProviderError"]


class ProviderError(Exception):
    """Raised only for setup problems (bad config, unreachable server on an explicit check).

    Per-request failures are *not* raised: they are returned as `Completion.error` so a single
    bad item cannot abort a long run.
    """


@runtime_checkable
class Provider(Protocol):
    """Turns a prompt into timed text."""

    @property
    def name(self) -> str:
        """Stable identifier recorded with every result, e.g. "ollama:llama3.1:8b".

        Declared read-only (a property, not a bare `name: str` attribute) so that
        implementations are free to compute it. A mutable protocol attribute would reject any
        implementation exposing `name` as a property -- which is exactly how OllamaProvider
        does it, and how a RAG or MLX provider naturally would too.
        """
        ...

    def complete(
        self, prompt: str, *, system: str | None = None, **overrides: Any
    ) -> Completion:
        """Generate a response, capturing timing.

        Must not raise for ordinary request failures -- return a Completion with `error` set.
        """
        ...

    def warmup(self) -> Completion:
        """Load weights before measurement begins, returning the throwaway generation.

        Mandatory in practice: measured on an M3, a cold call was 6.8x slower than warm, with
        93% of the difference being weight loading. Without warmup that outlier becomes the
        reported P95 (design §6.1).

        Returns a Completion rather than None so the Runner can persist the cold-start cost
        (flagged `is_warmup`) instead of discarding it. Each provider decides *how* to warm
        itself -- an HTTP model and a locally-loaded adapter differ -- while the caller still
        gets comparable data back.
        """
        ...

    def unload(self) -> None:
        """Release the model so the next one can be resident.

        On a 16GB machine two 8B models do not comfortably coexist; unloading between model
        batches avoids paying a 5-7s reload on every call (design §1.4).
        """
        ...
