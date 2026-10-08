"""Loading and integrity-checking the frozen golden set.

Stdlib only (design §4.1). Note what is *not* here: any dependency on HuggingFace
`datasets`. Building the golden set is a dev-time act (scripts/build_golden_set.py);
*reading* it at eval time only needs the committed JSONL, which keeps the runtime install
light and makes the eval set reproducible rather than re-downloaded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from viveka.core.types import GoldenItem

__all__ = [
    "GoldenSetError",
    "file_sha256",
    "load_golden_set",
    "load_meta",
    "meta_path_for",
]

REQUIRED_FIELDS = (
    "id",
    "source",
    "category",
    "question",
    "reference_answer",
    "grader",
)


class GoldenSetError(Exception):
    """Raised when the golden set is missing, malformed, or fails its hash check."""


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes, streamed."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def meta_path_for(jsonl_path: str | Path) -> Path:
    """data/golden_set.jsonl -> data/golden_set.meta.json"""
    p = Path(jsonl_path)
    return p.with_suffix("").with_suffix(".meta.json") if p.suffix else p


def load_meta(jsonl_path: str | Path) -> dict[str, Any]:
    meta_file = meta_path_for(jsonl_path)
    if not meta_file.exists():
        raise GoldenSetError(f"missing metadata sidecar: {meta_file}")
    with meta_file.open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def load_golden_set(
    path: str | Path = "data/golden_set.jsonl",
    *,
    verify_hash: bool = True,
    limit: int | None = None,
) -> list[GoldenItem]:
    """Read, validate and return the golden set.

    Args:
        path: the frozen JSONL.
        verify_hash: compare the file's SHA-256 against the value recorded in the sidecar.
            On by default. The CI gate is only meaningful if the eval set provably did not
            change between the baseline run and the candidate run -- without this check,
            deleting the three hardest items is enough to turn a red build green
            (design §7.2).
        limit: read at most N items (for fast smoke runs). Applied *after* validation, and
            it does not disable the hash check.

    Raises:
        GoldenSetError: on a missing file, schema violation, duplicate id, or hash mismatch.
    """
    jsonl = Path(path)
    if not jsonl.exists():
        raise GoldenSetError(
            f"golden set not found: {jsonl}. Build it with: python scripts/build_golden_set.py"
        )

    if verify_hash:
        meta = load_meta(jsonl)
        expected = meta.get("sha256")
        if not expected:
            raise GoldenSetError(f"sidecar {meta_path_for(jsonl)} has no 'sha256' field")
        actual = file_sha256(jsonl)
        if actual != expected:
            raise GoldenSetError(
                f"golden set hash mismatch for {jsonl}\n"
                f"  expected (sidecar): {expected}\n"
                f"  actual   (on disk): {actual}\n"
                "The eval set was modified without rebuilding its metadata. Comparing "
                "results across differing eval sets is invalid. Re-run "
                "scripts/build_golden_set.py, or restore the file."
            )

    items: list[GoldenItem] = []
    seen: dict[str, int] = {}
    with jsonl.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GoldenSetError(f"{jsonl}:{lineno}: invalid JSON: {exc}") from exc

            for fname in REQUIRED_FIELDS:
                value = raw.get(fname)
                if value is None or (isinstance(value, str) and not value.strip()):
                    raise GoldenSetError(
                        f"{jsonl}:{lineno}: missing or empty required field {fname!r}"
                    )

            item_id = raw["id"]
            if item_id in seen:
                raise GoldenSetError(
                    f"{jsonl}:{lineno}: duplicate id {item_id!r} (first seen on line "
                    f"{seen[item_id]}). Ids are the per-item join key across runs and must "
                    "be unique."
                )
            seen[item_id] = lineno
            items.append(GoldenItem.from_dict(raw))

    if not items:
        raise GoldenSetError(f"{jsonl} contains no items")

    return items[:limit] if limit is not None else items
