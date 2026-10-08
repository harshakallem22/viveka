"""Ollama provider -- the measurement core.

The only module in `viveka.core` that touches the network (design §4.1).

Streams `/api/chat` so that time-to-first-token can be measured client-side, and reads the
nanosecond counters from the stream's final chunk for server-side decode accounting. Both
clocks are needed; see §6.2-6.3 for why neither alone is sufficient.
"""

from __future__ import annotations

import contextlib
import json
import time
from typing import Any

import httpx

from viveka.core.types import Completion, Timing

__all__ = ["DEFAULT_BASE_URL", "OllamaProvider", "ollama_version"]

DEFAULT_BASE_URL = "http://localhost:11434"

#: Sent when we want a model evicted immediately.
_UNLOAD_KEEP_ALIVE = 0


def ollama_version(base_url: str = DEFAULT_BASE_URL, timeout: float = 5.0) -> str | None:
    """Server version, recorded as run provenance. None if unreachable."""
    try:
        resp = httpx.get(f"{base_url}/api/version", timeout=timeout)
        resp.raise_for_status()
        version: str | None = resp.json().get("version")
        return version
    except Exception:  # noqa: BLE001 - provenance only, never fatal
        return None


class OllamaProvider:
    """Calls one Ollama model.

    One instance per model. `options` are Ollama sampling options (temperature, seed,
    num_predict); `think` toggles reasoning traces on models that support them; `response_format`
    supplies a JSON schema for constrained decoding (used by the judge in Milestone 3).
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 120.0,
        keep_alive: str | int = "10m",
        options: dict[str, Any] | None = None,
        think: bool | None = None,
        response_format: dict[str, Any] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.keep_alive = keep_alive
        self.options = dict(options or {})
        self.think = think
        self.response_format = response_format
        self._owns_client = client is None
        # connect timeout stays short; read timeout must cover a slow generation.
        self._client = client or httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    # -- lifecycle ---------------------------------------------------------------------

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OllamaProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def warmup(self) -> Completion:
        """Force weight loading with a throwaway one-token generation.

        Returned rather than discarded so the caller can record `load_duration` -- the
        cold-start cost is a publishable number in its own right (design §6.1).
        """
        return self.complete("hi", num_predict=1)

    def unload(self) -> None:
        """Ask Ollama to evict this model now. Best-effort; never fatal."""
        # Eviction is an optimization, not a requirement: if it fails the next model simply
        # contends for RAM, which is slower but still correct.
        with contextlib.suppress(Exception):
            self._client.post(
                "/api/generate",
                json={"model": self.model, "keep_alive": _UNLOAD_KEEP_ALIVE},
                timeout=30.0,
            )

    # -- generation --------------------------------------------------------------------

    def _build_body(
        self, prompt: str, system: str | None, overrides: dict[str, Any]
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {**self.options, **overrides},
        }
        if self.think is not None:
            body["think"] = self.think
        if self.response_format is not None:
            body["format"] = self.response_format
        return body

    def complete(
        self, prompt: str, *, system: str | None = None, **overrides: Any
    ) -> Completion:
        """Generate, capturing client-side TTFT and server-side decode counters.

        Never raises for request failures -- the error is returned on the Completion so one bad
        item cannot abort a long run.
        """
        body = self._build_body(prompt, system, overrides)

        parts: list[str] = []
        ttft_ms: float | None = None
        chunk_count = 0
        final: dict[str, Any] = {}
        error: str | None = None

        start = time.perf_counter()
        try:
            with self._client.stream("POST", "/api/chat", json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise RuntimeError(str(chunk["error"]))
                    chunk_count += 1

                    content = (chunk.get("message") or {}).get("content") or ""
                    if content:
                        # TTFT is stamped on the first chunk carrying *visible* content.
                        # Reasoning models stream a `thinking` field first; counting that
                        # would report a flattering TTFT for text the user never sees (§6.2).
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - start) * 1000.0
                        parts.append(content)

                    if chunk.get("done"):
                        final = chunk
        except Exception as exc:  # noqa: BLE001 - deliberately broad: see docstring
            error = f"{type(exc).__name__}: {exc}"

        total_ms = (time.perf_counter() - start) * 1000.0
        text = "".join(parts)

        if error is not None:
            return Completion(
                text=text,
                timing=Timing(total_ms=total_ms, ttft_ms=ttft_ms),
                raw={"chunks": chunk_count},
                error=error,
            )

        return Completion(
            text=text,
            timing=Timing.from_ollama(final, total_ms=total_ms, ttft_ms=ttft_ms),
            raw={
                "chunks": chunk_count,
                # "length" means num_predict truncated the response. Critical to keep: a
                # truncated answer scores incorrect for a harness reason, not a model reason.
                "done_reason": final.get("done_reason"),
                "final": final,
            },
        )
