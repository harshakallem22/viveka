"""Providers -- implementations of the Provider protocol.

`base` is stdlib-only; `ollama` is the sole network-touching module in viveka.core.
"""

from viveka.core.providers.base import Provider, ProviderError
from viveka.core.providers.ollama import DEFAULT_BASE_URL, OllamaProvider, ollama_version

__all__ = [
    "DEFAULT_BASE_URL",
    "OllamaProvider",
    "Provider",
    "ProviderError",
    "ollama_version",
]
