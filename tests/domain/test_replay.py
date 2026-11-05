"""BattleReplay: the repair is applied before any feature and never sees the label."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from data import DataError, Dataset
from domain.battle import BattleContext, BattleReplay, TurnState
from domain.errors import ReplayError, UnknownSpeciesError
from domain.pokedex import Missing, Pokedex, PokedexBuilder
from domain.repair import NoRepair

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "battles.jsonl"

Record = dict[str, Any]


class SpyRepair:
    """Marks the given rows as faints of whatever P2 had in front, and remembers its input."""

    def __init__(self, rows: tuple[int, ...] = ()) -> None:
        self.rows = rows
        self.seen: list[Mapping[str, Any]] = []

    def probabilities(self, record: Mapping[str, Any]) -> Mapping[int, float]:
        self.seen.append(record)
        return {}

    def repairs(self, record: Mapping[str, Any]) -> Mapping[int, str]:
        self.seen.append(record)
        timeline = record["battle_timeline"]
        return {
            i: timeline[i - 1]["p2_pokemon_state"]["name"] for i in self.rows if i < len(timeline)
        }

    def describe(self) -> dict[str, Any]:
        return {"name": "spy", "version": "1", "params": {}}


@pytest.fixture(scope="module")
def records() -> list[Record]:
    return [json.loads(line) for line in FIXTURES.open()]


@pytest.fixture(scope="module")
def pokedex(records: list[Record]) -> Pokedex:
    return PokedexBuilder().add_records(records).build()


def first_p2_switch(record: Record) -> int | None:
    """Row of P2's first replacement of a Pokémon not recorded as fainted that never returns."""
    names = [t["p2_pokemon_state"]["name"] for t in record["battle_timeline"]]
    for i in range(1, len(names)):
        status = record["battle_timeline"][i - 1]["p2_pokemon_state"]["status"]
        if names[i - 1] != names[i] and status != "fnt" and names[i - 1] not in names[i:]:
            return i
    return None


def states(context: BattleContext) -> list[TurnState]:
    return list(context.history)


def test_replay_is_deterministic(records: list[Record], pokedex: Pokedex) -> None:
    replayer = BattleReplay(pokedex, unknown=Missing())
    assert states(replayer.replay(records[0])) == states(replayer.replay(records[0]))


def test_default_repair_keeps_the_timeline_as_recorded(
    records: list[Record], pokedex: Pokedex
) -> None:
    context = BattleReplay(pokedex, unknown=Missing()).replay(records[0])
    assert isinstance(context.repair, NoRepair)
    assert context.repair.probabilities(context.record) == {}
    assert context.repair.repairs(context.record) == {}
    assert json.loads(json.dumps(context.repair.describe()))["name"] == "none"


def test_the_repair_never_sees_the_label(records: list[Record], pokedex: Pokedex) -> None:
    record = next(r for r in records if "player_won" in r)
    spy = SpyRepair()
    context = BattleReplay(pokedex, spy, Missing()).replay(record)
    context.at_turn(context.n_turns - 1)
    assert len(spy.seen) == 2
    assert all("player_won" not in seen for seen in spy.seen)
    assert "player_won" not in context.record
    assert "player_won" in record


def test_repairs_reach_the_history_and_the_final_state(
    records: list[Record], pokedex: Pokedex
) -> None:
    record = next(r for r in records if first_p2_switch(r) is not None)
    row = first_p2_switch(record)
    assert row is not None
    leaver = record["battle_timeline"][row - 1]["p2_pokemon_state"]["name"]
    plain = BattleReplay(pokedex, unknown=Missing()).replay(record)
    repaired = BattleReplay(pokedex, SpyRepair((row,)), Missing()).replay(record)
    assert repaired.history[row].p2.n_fainted == plain.history[row].p2.n_fainted + 1
    assert repaired.history[row - 1].p2 == plain.history[row - 1].p2
    assert repaired.side("p2").team[leaver].fainted
    assert repaired.history[row].p2.voluntary_switch is False
    assert plain.history[row].p2.voluntary_switch is True


def test_at_turn_replays_a_truncated_timeline_through_the_repair(
    records: list[Record], pokedex: Pokedex
) -> None:
    record = next(r for r in records if first_p2_switch(r) is not None)
    row = first_p2_switch(record)
    assert row is not None
    spy = SpyRepair((row,))
    context = BattleReplay(pokedex, spy, Missing()).replay(record)
    early = context.at_turn(row)
    assert early.n_turns == row
    assert early.history == context.history[:row]
    assert len(spy.seen[-1]["battle_timeline"]) == row
    assert early.repair is spy
    assert context.at_turn(None) is context
    assert context.at_turn(context.n_turns) is context


def test_strict_refuses_and_missing_keeps_a_species_without_stats(
    records: list[Record], pokedex: Pokedex
) -> None:
    record = json.loads(json.dumps(records[0]))
    record["battle_timeline"][-1]["p2_pokemon_state"]["name"] = "hidden"
    with pytest.raises(UnknownSpeciesError):
        BattleReplay(pokedex).replay(record)
    assert BattleReplay(pokedex, unknown=Missing()).replay(record).last.p2.active == "hidden"


def test_an_empty_timeline_cannot_be_replayed(records: list[Record], pokedex: Pokedex) -> None:
    empty = {**records[0], "battle_timeline": []}
    with pytest.raises(ReplayError, match="empty timeline"):
        BattleReplay(pokedex, unknown=Missing()).replay(empty)
    with pytest.raises(DataError, match="empty timeline"):
        Dataset.from_records([empty], pokedex)
