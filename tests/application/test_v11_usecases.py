"""``train`` and ``evaluate`` refuse ``--test`` with v11 (its features fit on training battles)."""

from __future__ import annotations

from pathlib import Path

import pytest

import usecases
from settings import Settings

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "battles.jsonl"


@pytest.mark.parametrize("command", [usecases.train, usecases.evaluate])
def test_v11_refuses_unlabelled_test_battles(command: object) -> None:
    settings = Settings(FIXTURES, test=FIXTURES, repair="none")
    with pytest.raises(SystemExit, match="drop --test"):
        command(settings, ["hp30", "v11"], ["logreg"])  # type: ignore[operator]
