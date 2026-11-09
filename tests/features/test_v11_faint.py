"""v11's hidden-faint detector on the fixtures: train on 150 battles, score the other 50."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from features.core.v11.faint import fit_detector
from features.core.v11.mechanics import Pokedex
from features.core.v11.replay import Battle, parse

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "battles.jsonl"
N_TRAIN = 150

Record = dict[str, Any]


def detect(records: list[Record]) -> tuple[list[Battle], list[Battle]]:
    dex = Pokedex(records[:N_TRAIN])
    train, test = parse(records[:N_TRAIN], dex), parse(records[N_TRAIN:], dex)
    fit_detector(train, test)
    return train, test


@pytest.fixture(scope="module")
def records() -> list[Record]:
    return [json.loads(line) for line in FIXTURES.open()]


@pytest.fixture(scope="module")
def detected(records: list[Record]) -> tuple[list[Battle], list[Battle]]:
    return detect(records)


def test_recorded_faints_get_one_and_candidates_a_probability(
    detected: tuple[list[Battle], list[Battle]],
) -> None:
    recorded = candidates = 0
    for battles in detected:
        for battle in battles:
            rows = battle.transitions("p2")
            assert set(battle.p2_faint) == {row["i"] for row in rows}
            for row in rows:
                p = battle.p2_faint[row["i"]]
                if row["recorded"]:
                    assert p == 1.0
                    recorded += 1
                elif row["final"]:
                    assert 0.0 <= p <= 1.0
                    candidates += 1
                else:
                    assert p == 0.0
    assert recorded > 0 and candidates > 0


def test_the_detector_is_deterministic(
    records: list[Record], detected: tuple[list[Battle], list[Battle]]
) -> None:
    again = detect(records)
    for first, second in zip(detected, again, strict=True):
        assert [b.p2_faint for b in first] == [b.p2_faint for b in second]
