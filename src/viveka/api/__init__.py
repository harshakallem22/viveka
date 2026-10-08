"""FastAPI layer. Optional extra: `pip install viveka[api]`.

Imported by nothing in `viveka.core` -- the dependency runs one way only (constraint C1).
"""

from viveka.api.main import app

__all__ = ["app"]
