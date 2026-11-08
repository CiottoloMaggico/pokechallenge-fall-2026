"""Dependency rules of REFACTORING.md §1, checked on the imports of every module (ast).

* ``domain`` imports only the standard library and numpy (so never an addon, a feature,
  a model or the application);
* ``addons`` never import the application or the CLI;
* ``features`` (and ``feature_set.py``) never import models, addons or the application;
* ``models`` never import the domain, features, addons or the application;
* the CLI (``main.py``) only parses arguments and calls a use case: it never runs the gym,
  the tuner, the selection or loads the data itself.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APPLICATION = {"gym", "tuning", "usecases", "detector", "selection", "main", "folds", "settings"}
DOMAIN_ALLOWED = {"domain", "numpy", *sys.stdlib_module_names}


def modules(*parts: str) -> list[Path]:
    """Python files of the packages or modules ``parts`` (relative to the root)."""
    found: list[Path] = []
    for part in parts:
        path = ROOT / part
        found.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    return found


FORBIDDEN: dict[str, tuple[list[Path], set[str]]] = {
    "addons": (modules("addons"), APPLICATION),
    "features": (modules("features", "feature_set.py"), {"models", "addons", *APPLICATION}),
    "models": (modules("models"), {"domain", "features", "feature_set", "addons", *APPLICATION}),
    "cli": (modules("main.py"), {"gym", "tuning", "selection", "data", "stats_tables"}),
}
CASES = [
    pytest.param(path, banned, id=f"{layer}:{path.relative_to(ROOT)}")
    for layer, (paths, banned) in FORBIDDEN.items()
    for path in paths
]


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


def test_every_layer_has_modules_to_check() -> None:
    assert {"battle.py", "repair.py", "transitions.py"} <= {p.name for p in modules("domain")}
    assert {"knockouts_model.py", "knockouts_rule.py"} <= {p.name for p in modules("addons")}
    assert {"compact_v5.py", "material.py", "feature_set.py"} <= {
        p.name for p in FORBIDDEN["features"][0]
    }
    assert {"logreg.py", "estimator.py", "logit_blend.py"} <= {p.name for p in modules("models")}


@pytest.mark.parametrize("path", modules("domain"), ids=lambda p: p.name)
def test_domain_imports_only_stdlib_and_numpy(path: Path) -> None:
    assert imported_roots(path.read_text(encoding="utf-8")) <= DOMAIN_ALLOWED


@pytest.mark.parametrize(("path", "banned"), CASES)
def test_a_layer_never_imports_what_it_must_not(path: Path, banned: set[str]) -> None:
    assert not imported_roots(path.read_text(encoding="utf-8")) & banned
