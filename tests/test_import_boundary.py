"""Mechanical enforcement of the core/ decoupling rule (constraint C1).

`viveka.core` is the part Projects 2 and 4 import. If it ever grows a dependency on the API,
CLI, or dashboard layers, the reuse story quietly breaks -- and folder separation alone is a
convention that erodes under pressure. So it is a test.

Walks the AST rather than importing, so a violation is reported as a clean failure instead of
an ImportError, and so it catches imports inside functions too.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = REPO_ROOT / "src" / "viveka" / "core"

#: viveka.core must never reach for these.
FORBIDDEN_IN_CORE = {
    "fastapi",
    "uvicorn",
    "starlette",
    "datasets",       # dev-time golden-set building only; the runtime reads committed JSONL
    "huggingface_hub",
    "viveka.api",
    "viveka.cli",
}

#: Modules held to a stricter rule: standard library only (design §4.1). These are the
#: numerical and persistence heart of the library, and keeping them dependency-free is what
#: makes them trivially testable and portable.
STDLIB_ONLY_MODULES = {
    "types.py",
    "metrics.py",
    "agreement.py",
    "store.py",
    "datasets.py",
}


def core_modules() -> list[Path]:
    return sorted(CORE_DIR.rglob("*.py"))


def imported_names(path: Path) -> list[tuple[str, int]]:
    """Every top-level module name imported by this file, with line numbers."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import, stays inside the package
                continue
            if node.module:
                found.append((node.module, node.lineno))
    return found


def test_core_directory_is_populated():
    """Guard against the suite silently passing because it found no files."""
    assert core_modules(), f"no modules found under {CORE_DIR}"


@pytest.mark.parametrize("module_path", core_modules(), ids=lambda p: p.name)
def test_core_does_not_import_api_or_dashboard_layers(module_path: Path):
    violations = []
    for name, lineno in imported_names(module_path):
        root = name.split(".")[0]
        if name in FORBIDDEN_IN_CORE or root in FORBIDDEN_IN_CORE:
            violations.append(f"{module_path.name}:{lineno} imports {name!r}")
    assert not violations, (
        "viveka.core must stay importable without the API/dashboard/dataset layers "
        "(constraint C1):\n  " + "\n  ".join(violations)
    )


@pytest.mark.parametrize(
    "module_path",
    [p for p in core_modules() if p.name in STDLIB_ONLY_MODULES],
    ids=lambda p: p.name,
)
def test_designated_modules_use_only_the_standard_library(module_path: Path):
    third_party = []
    for name, lineno in imported_names(module_path):
        root = name.split(".")[0]
        if root == "viveka" or root in sys.stdlib_module_names:
            continue
        third_party.append(f"{module_path.name}:{lineno} imports {name!r}")
    assert not third_party, (
        f"{module_path.name} is designated standard-library-only (design §4.1):\n  "
        + "\n  ".join(third_party)
    )


def test_core_is_importable_without_optional_extras():
    """Importing viveka.core must pull in nothing from the api/data extras.

    Runs in a subprocess: a fresh interpreter is the only honest way to check what an import
    actually drags in, and it avoids mutating this process's sys.modules. Catches transitive
    dependencies that the AST scan above cannot see.
    """
    import subprocess

    probe = (
        "import sys, viveka.core;"
        "leaked = {'fastapi','uvicorn','datasets','pyarrow','pandas'} & set(sys.modules);"
        "print(','.join(sorted(leaked)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, f"importing viveka.core failed:\n{result.stderr}"
    leaked = result.stdout.strip()
    assert not leaked, f"importing viveka.core pulled in optional extras: {leaked}"
