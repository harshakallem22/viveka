"""ReplayProvider -- serves recorded model responses instead of calling a model.

Stdlib only. This is what makes Tier-1 CI possible (design §11.2): the entire pipeline --
run, judge, report, gate -- executes deterministically in seconds on a hosted runner with no GPU
and no models installed. Only the network call is substituted; the orchestration, metrics,
extraction and gate logic under test are the real ones.

Cassettes hold **real recorded output**, generated from a genuine benchmark run by
`scripts/record_cassettes.py`, not hand-written fakes. A hand-written fixture would only ever
exercise the paths its author remembered.

Matching is by SHA-256 of the prompt. Prompts are deterministic functions of the golden set and
the prompt template, so an exact hash match is achievable -- and if a prompt template changes, the
cassette misses loudly instead of silently replaying the wrong answer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from viveka.core.types import Completion, Timing

__all__ = ["CassetteMiss", "ReplayProvider", "prompt_key"]


class CassetteMiss(LookupError):
    """No recording for this prompt. Deliberately loud -- see module docstring."""


def prompt_key(prompt: str) -> str:
    """Stable cassette key. Whitespace-normalized so trivial formatting drift does not miss."""
    normalized = "\n".join(line.rstrip() for line in prompt.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


class ReplayProvider:
    """Replays recorded completions for one model.

    Args:
        name: the provider name to report, e.g. "ollama:gemma3:4b". Kept identical to the recorded
            model so replayed results are directly comparable to live ones.
        entries: mapping of prompt key -> recorded payload.
        strict: raise `CassetteMiss` on an unknown prompt. Leave True in CI -- a silent fallback
            would let a prompt change pass tests while producing nonsense.
    """

    def __init__(
        self,
        name: str,
        entries: dict[str, dict[str, Any]],
        *,
        strict: bool = True,
    ) -> None:
        self._name = name
        self.entries = entries
        self.strict = strict
        self.calls = 0
        self.misses: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def from_file(cls, path: str | Path, *, strict: bool = True) -> ReplayProvider:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["model"], data["entries"], strict=strict)

    @classmethod
    def load_all(cls, directory: str | Path, *, strict: bool = True) -> list[ReplayProvider]:
        """Load every cassette in a directory, one provider per file, sorted for determinism."""
        return [
            cls.from_file(p, strict=strict)
            for p in sorted(Path(directory).glob("*.json"))
            if p.name != "judge.json"
        ]

    def complete(self, prompt: str, *, system: str | None = None, **overrides: Any) -> Completion:
        self.calls += 1
        key = prompt_key(prompt)
        payload = self.entries.get(key)
        if payload is None:
            self.misses.append(key)
            if self.strict:
                raise CassetteMiss(
                    f"{self._name}: no recording for prompt {key} "
                    f"({len(self.entries)} recorded). Re-record with "
                    f"scripts/record_cassettes.py after changing a prompt template.\n"
                    f"--- prompt begins ---\n{prompt[:400]}"
                )
            return Completion(
                text="", timing=Timing(total_ms=0.0), error=f"CassetteMiss: {key}"
            )

        timing = Timing(
            total_ms=payload.get("total_ms", 0.0),
            ttft_ms=payload.get("ttft_ms"),
            load_ms=payload.get("load_ms"),
            prompt_eval_ms=payload.get("prompt_eval_ms"),
            decode_ms=payload.get("decode_ms"),
            prompt_tokens=payload.get("prompt_tokens"),
            output_tokens=payload.get("output_tokens"),
        )
        return Completion(
            text=payload.get("text", ""),
            timing=timing,
            raw={"done_reason": payload.get("done_reason"), "replayed": True},
            error=payload.get("error"),
        )

    def warmup(self) -> Completion:
        """Synthetic cold start, so warmup-exclusion logic is still exercised in CI."""
        return Completion(
            text="",
            timing=Timing(total_ms=5000.0, load_ms=4800.0),
            raw={"done_reason": "stop", "replayed": True},
        )

    def unload(self) -> None:
        """No-op: nothing is resident."""

    def close(self) -> None:
        """No-op, for interface parity with OllamaProvider."""
