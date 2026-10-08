"""Configuration loading and validation.

Lives outside `core/` because it depends on pydantic and yaml, and because `core` should be
usable by a caller who constructs objects directly rather than reading our config file.

`extra="forbid"` throughout, deliberately: a typo'd `temperture: 0` must fail loudly. Silently
ignoring an unknown key would mean publishing benchmark numbers produced by settings the
operator did not intend.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["Config", "ContestantConfig", "GatesConfig", "JudgeConfig", "RunSettings", "load_config"]

DEFAULT_CONFIG_PATH = "viveka.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SamplingDefaults(_Strict):
    temperature: float = 0.0
    seed: int = 7
    #: Response cap. Too low truncates chain-of-thought and scores a correct model as wrong;
    #: `done_reason == "length"` in the results is the signal that this needs raising.
    num_predict: int = 512


class RunSettings(_Strict):
    golden_set: str = "data/golden_set.jsonl"
    db: str = "data/viveka.db"
    repeats: int = Field(default=1, ge=1)
    warmup: bool = True
    resume: bool = True
    #: Sequential by design: 16GB cannot host two 8B models at once (design C4).
    concurrency: int = Field(default=1, ge=1, le=1)
    keep_alive: str | int = "10m"
    timeout_s: float = Field(default=120.0, gt=0)
    base_url: str = "http://localhost:11434"
    defaults: SamplingDefaults = Field(default_factory=SamplingDefaults)

    @field_validator("concurrency")
    @classmethod
    def _sequential_only(cls, v: int) -> int:
        if v != 1:
            raise ValueError(
                "concurrency must be 1: parallel model calls contend for 16GB of RAM and "
                "make latency percentiles meaningless (design C4)"
            )
        return v


class ContestantConfig(_Strict):
    """One model under test. Adding a contestant is one line of YAML (design C5)."""

    model: str
    #: Per-model sampling overrides, if a model needs different settings.
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model")
    @classmethod
    def _normalize_tag(cls, v: str) -> str:
        """`mistral` -> `mistral:latest`.

        Ollama resolves the bare name implicitly, but recording it explicitly means results
        from months apart are not silently comparing different model builds.
        """
        v = v.strip()
        return v if ":" in v else f"{v}:latest"


class JudgeConfig(_Strict):
    model: str = "qwen3:8b"
    #: qwen3 emits a <think> block by default: slower, and it makes judge cost unpredictable.
    #: Verified that `think: false` suppresses it (design §1.3).
    think: bool = False
    temperature: float = 0.0
    seed: int = 7
    num_predict: int = 400
    prompt_version: str = "v1"

    @field_validator("model")
    @classmethod
    def _normalize_tag(cls, v: str) -> str:
        v = v.strip()
        return v if ":" in v else f"{v}:latest"


class GatesConfig(_Strict):
    """CI thresholds. Start permissive; tighten after measuring the noise floor (design §11.3)."""

    min_accuracy: float = Field(default=0.55, ge=0.0, le=1.0)
    max_p95_latency_ms: float = Field(default=45000.0, gt=0)
    max_accuracy_drop: float = Field(default=0.10, ge=0.0, le=1.0)
    #: Item-level gating is more sensitive and more interpretable than an aggregate delta.
    max_regressed_items: int = Field(default=3, ge=0)
    require_same_golden_set: bool = True


class Config(_Strict):
    run: RunSettings = Field(default_factory=RunSettings)
    contestants: list[ContestantConfig] = Field(default_factory=list)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    gates: GatesConfig = Field(default_factory=GatesConfig)

    @field_validator("contestants")
    @classmethod
    def _no_duplicates(cls, v: list[ContestantConfig]) -> list[ContestantConfig]:
        names = [c.model for c in v]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate contestants: {sorted(dupes)}")
        return v

    def snapshot(self) -> dict[str, Any]:
        """Fully resolved config, stored per-run so a result is always self-describing."""
        return self.model_dump(mode="json")

    def config_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.snapshot(), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def judge_is_a_contestant(self) -> bool:
        """A judge that is also under test would grade its own output (self-preference bias)."""
        return self.judge.model in {c.model for c in self.contestants}


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate viveka.yaml. Missing file yields defaults."""
    p = Path(path)
    if not p.exists():
        return Config()
    with p.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: expected a mapping at the top level, got {type(raw).__name__}")
    return Config.model_validate(raw)
