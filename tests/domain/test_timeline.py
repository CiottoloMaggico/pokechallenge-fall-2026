"""The observed timeline on the fixtures: columns as recorded, events, revealed moves, Spikes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from domain.timeline import ObservedTimeline

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "battles.jsonl"

Record = dict[str, Any]


@pytest.fixture(scope="module")
def records() -> list[Record]:
    return [json.loads(line) for line in FIXTURES.open()]


@pytest.fixture(scope="module")
def timelines(records: list[Record]) -> list[ObservedTimeline]:
    return [ObservedTimeline(r) for r in records]


def test_the_columns_are_the_records(records: list[Record]) -> None:
    record = records[0]
    timeline = ObservedTimeline(record)
    for i, turn in enumerate(record["battle_timeline"]):
        state = turn["p2_pokemon_state"]
        side = timeline.sides["p2"]
        assert side.names[i] == state["name"] and side.hp[i] == state["hp_pct"]
        assert side.status[i] == state["status"] and side.boosts[i] is state["boosts"]
        assert side.effects[i] == frozenset(state["effects"]) - {"noeffect"}
        assert side.moves[i] is turn["p2_move_details"]
    assert timeline.n == len(record["battle_timeline"]) == timeline.sides["p1"].n


def test_a_repeated_move_is_one_event(timelines: list[ObservedTimeline]) -> None:
    repeats = 0
    for timeline in timelines:
        for side in timeline.sides.values():
            for i in range(1, side.n):
                now, before = side.moves[i], side.moves[i - 1]
                same = side.names[i] == side.names[i - 1]
                if same and now is not None and before is not None:
                    fresh = now["name"] != before["name"]
                    assert (side.events[i] is not None) == fresh
                    repeats += not fresh
    assert repeats > 0, "the fixtures must cover a repeated move"


def test_revealed_moves_and_the_p2_lead(records: list[Record]) -> None:
    for record in records:
        timeline = ObservedTimeline(record)
        p2 = timeline.sides["p2"]
        assert record["p2_lead_details"]["name"] in p2.revealed
        for name, moves in p2.revealed.items():
            used = {m["name"] for n, m in zip(p2.names, p2.moves, strict=True) if n == name and m}
            assert set(moves) == used
        for name, first in p2.first_index.items():
            assert p2.names[first] == name and name not in p2.names[:first]
            assert name not in p2.names[p2.last_index[name] + 1 :]


def test_spikes_laid_and_spun(timelines: list[ObservedTimeline]) -> None:
    laid = spun = 0
    for timeline in timelines:
        for own, opp in (("p1", "p2"), ("p2", "p1")):
            spikes = timeline.spikes[own]
            for i in range(1, timeline.n):
                set_now = timeline.sides[opp].events[i]
                spin_now = timeline.sides[own].events[i]
                if spin_now is not None and spin_now["name"] == "rapidspin":
                    assert not spikes[i]
                    spun += spikes[i - 1]
                elif set_now is not None and set_now["name"] == "spikes":
                    assert spikes[i]
                    laid += not spikes[i - 1]
                else:
                    assert spikes[i] == spikes[i - 1]
    assert laid > 0 and spun > 0, "the fixtures must lay and spin Spikes"


def test_used_move_skips_switch_ins_sleep_and_replacements(
    timelines: list[ObservedTimeline],
) -> None:
    skipped = 0
    for timeline in timelines:
        p1 = timeline.sides["p1"]
        for s, side in timeline.sides.items():
            assert timeline.used_move(s, 0) is side.moves[0]
            for i in range(1, timeline.n):
                stuck = side.status[i - 1] in ("slp", "frz") and side.status[i] in ("slp", "frz")
                switched = side.names[i] != side.names[i - 1]
                if p1.status[i - 1] == "fnt" or switched or stuck:
                    assert timeline.used_move(s, i) is None
                    skipped += side.moves[i] is not None
                else:
                    assert timeline.used_move(s, i) is side.moves[i]
    assert skipped > 0


def test_levels_and_items(records: list[Record]) -> None:
    record = records[0]
    timeline = ObservedTimeline(record)
    first = record["p1_team_details"][0]
    lead = record["p2_lead_details"]
    assert timeline.level("p1", first["name"]) == first["level"]
    assert timeline.item("p1", first["name"]) == first["item"]
    assert timeline.level("p2", lead["name"]) == lead["level"]
    assert timeline.item("p2", lead["name"]) is None
    assert timeline.level("p2", "not-shown") == 100
