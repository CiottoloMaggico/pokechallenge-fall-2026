"""Dependency rules of REFACTORING.md §1: ``domain`` imports only the stdlib and numpy,
``addons`` never import the application or the CLI."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOMAIN_MODULES = sorted((ROOT / "domain").rglob("*.py"))
ALLOWED = {"domain", "numpy", *sys.stdlib_module_names}
ADDON_MODULES = sorted((ROOT / "addons").rglob("*.py"))
APPLICATION = {"gym", "tuning", "usecases", "detector", "selection", "main", "folds", "settings"}


def imported_roots(source: str) -> set[str]:
    """Top-level packages imported anywhere in ``source``, deferred imports included.

    Relative imports stay inside the package, so they have no root to check.
    """
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.partition(".")[0])
    return roots


def test_the_scan_sees_every_form_of_import() -> None:
    source = (
        "import os.path, pandas as pd\n"
        "from sklearn.metrics import log_loss\n"
        "from . import sibling\n"
        "def later():\n"
        "    import gym\n"
        "    from addons.rule import Rule\n"
    )
    assert imported_roots(source) == {"os", "pandas", "sklearn", "gym", "addons"}


def test_there_are_domain_modules_to_check() -> None:
    assert {"battle.py", "repair.py", "transitions.py"} <= {p.name for p in DOMAIN_MODULES}


@pytest.mark.parametrize("path", DOMAIN_MODULES, ids=lambda p: p.name)
def test_domain_imports_only_stdlib_and_numpy(path: Path) -> None:
    assert imported_roots(path.read_text(encoding="utf-8")) <= ALLOWED


def test_there_are_addons_to_check() -> None:
    assert {"knockouts_model.py", "knockouts_rule.py"} <= {p.name for p in ADDON_MODULES}


@pytest.mark.parametrize("path", ADDON_MODULES, ids=lambda p: p.name)
def test_addons_never_import_the_application(path: Path) -> None:
    assert not imported_roots(path.read_text(encoding="utf-8")) & APPLICATION
