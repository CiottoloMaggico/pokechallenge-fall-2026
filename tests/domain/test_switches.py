"""Switch descriptions on the fixtures: P1 faints read from the entry before, P2 candidates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from domain.pokedex import Missing, PokedexBuilder
from domain.switches import COLUMNS, SwitchExtractor
from domain.timeline import ObservedTimeline
from domain.typechart import TypeChart

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "battles.jsonl"
CHART = TypeChart.from_csv(ROOT / "assets" / "type_chart_gen2.csv")

Record = dict[str, Any]


@pytest.fixture(scope="module")
def records() -> list[Record]:
    return [json.loads(line) for line in FIXTURES.open()]


def extractor(records: list[Record], record: Record) -> SwitchExtractor:
    pokedex = PokedexBuilder().add_records(records).build()
    return SwitchExtractor(pokedex.lookup(record, Missing()), CHART)


def test_every_change_of_active_is_described_in_order(records: list[Record]) -> None:
    for record in records[:50]:
        timeline = ObservedTimeline(record)
        for side in ("p1", "p2"):
            names = timeline.sides[side].names
            changes = [i for i in range(1, timeline.n) if names[i] != names[i - 1]]
            switches = extractor(records, record).extract(record, side)
            assert [s.index for s in switches] == changes
            assert all(s.leaver == names[s.index - 1] for s in switches)
            assert all(tuple(s.features) in ((), COLUMNS) for s in switches)


def test_a_p1_faint_skips_the_fnt_entry(records: list[Record]) -> None:
    found = 0
    for record in records:
        timeline = ObservedTimeline(record)
        own, opp = timeline.sides["p1"], timeline.sides["p2"]
        for switch in extractor(records, record).extract(record, "p1"):
            i = switch.index
            if own.status[i - 1] != "fnt":
                assert not switch.recorded and switch.label == 0
                assert switch.final == (own.last_index[switch.leaver] == i - 1)
                continue
            assert switch.recorded and switch.label == 1
            usable = opp.status[i - 1] != "fnt" and i >= 2 and own.names[i - 2] == switch.leaver
            assert bool(switch.features) == usable
            if usable:
                assert switch.features["x_hp"] == own.hp[i - 2] and switch.features["turn"] == i
                found += 1
            else:
                assert switch.final
    assert found > 0, "the fixtures must cover a P1 faint"


def test_p2_faints_are_never_described(records: list[Record]) -> None:
    for record in records:
        for switch in extractor(records, record).extract(record, "p2"):
            assert switch.recorded == (not switch.features)


def test_examples_and_candidates(records: list[Record]) -> None:
    examples = candidates = 0
    for record in records:
        found = extractor(records, record)
        rows = found.examples(record)
        assert all(s.final and s.features and not math.isnan(s.features["x_hp"]) for s in rows)
        cands = found.candidates(record)
        assert all(s.final and not s.recorded and s.features for s in cands)
        examples, candidates = examples + len(rows), candidates + len(cands)
    assert examples > 0 and candidates > 0


def test_values_keep_their_types(records: list[Record]) -> None:
    for record in records[:20]:
        for switch in extractor(records, record).candidates(record):
            values = switch.features
            assert isinstance(values["turn"], int) and isinstance(values["x_slp"], bool)
            assert isinstance(values["a_dmg"], float)
            assert values["x_hp_minus_dmg"] == values["x_hp"] - values["a_dmg"]
