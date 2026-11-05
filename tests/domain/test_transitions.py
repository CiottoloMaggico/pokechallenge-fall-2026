"""One test per kind of switch, on real battles of the fixtures."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from domain.damage import DamageCalculator
from domain.pokedex import PokedexBuilder
from domain.records import without_label
from domain.transitions import ALL_COLUMNS, Transition, TransitionExtractor
from domain.typechart import TypeChart

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "battles.jsonl"
SIDES = (("p1", "p2"), ("p2", "p1"))

Record = dict[str, Any]
Case = tuple[Record, Transition, str, str]


@pytest.fixture(scope="module")
def records() -> list[Record]:
    return [json.loads(line) for line in FIXTURES.open()]


@pytest.fixture(scope="module")
def extractor(records: list[Record]) -> TransitionExtractor:
    pokedex = PokedexBuilder().add_records(records).build()
    chart = TypeChart.from_csv(ROOT / "assets" / "type_chart_gen2.csv")
    return TransitionExtractor(pokedex.find, DamageCalculator(chart))


@pytest.fixture(scope="module")
def cases(records: list[Record], extractor: TransitionExtractor) -> list[Case]:
    """Every transition of both sides of every battle."""

    def of(record: Record) -> Iterator[Case]:
        for side, other in SIDES:
            for transition in extractor.extract(record, side):  # type: ignore[arg-type]
                yield record, transition, side, other

    return [case for record in records for case in of(record)]


def state(record: Record, row: int, side: str) -> dict[str, Any]:
    return record["battle_timeline"][row][f"{side}_pokemon_state"]


def move_name(record: Record, row: int, side: str) -> str | None:
    move = record["battle_timeline"][row][f"{side}_move_details"]
    return move["name"] if move else None


def matching(cases: list[Case], condition: Callable[[Case], bool]) -> list[Case]:
    found = [case for case in cases if condition(case)]
    assert found, "the fixtures must cover this case"
    return found


def test_p1_faint_skips_the_fnt_row(cases: list[Case]) -> None:
    faints = matching(
        cases,
        lambda c: (
            c[2] == "p1"
            and c[1].fainted
            and state(c[0], c[1].index - 1, "p2")["status"] != "fnt"
            and c[1].index >= 2
            and state(c[0], c[1].index - 2, "p1")["name"] == c[1].leaver
        ),
    )
    for record, transition, _, _ in faints:
        i = transition.index
        assert state(record, i - 1, "p1")["status"] == "fnt"
        assert state(record, i - 1, "p1")["hp_pct"] == 0
        assert transition.features["x_hp"] == state(record, i - 2, "p1")["hp_pct"]
        assert transition.features["y_hp"] == state(record, i, "p1")["hp_pct"]
        assert transition.features["turn"] == i
        assert transition.final


def test_double_knockout_carries_no_features(cases: list[Case]) -> None:
    doubles = matching(
        cases,
        lambda c: (
            state(c[0], c[1].index - 1, "p1")["status"] == "fnt"
            and state(c[0], c[1].index - 1, "p2")["status"] == "fnt"
        ),
    )
    assert {side for _, _, side, _ in doubles} == {"p1", "p2"}
    for _, transition, _, _ in doubles:
        assert transition.fainted
        assert transition.features == {}


def test_recorded_p2_faint_is_never_described(cases: list[Case]) -> None:
    for _, transition, _, _ in matching(cases, lambda c: c[2] == "p2" and c[1].fainted):
        assert transition.features == {}


def test_voluntary_switch_reads_the_row_before(cases: list[Case]) -> None:
    switches = matching(cases, lambda c: not c[1].fainted)
    for record, transition, side, _ in switches:
        i = transition.index
        assert transition.leaver == state(record, i - 1, side)["name"]
        assert transition.features["x_hp"] == state(record, i - 1, side)["hp_pct"]
        assert transition.features["turn"] == i
        assert sorted(transition.features) == sorted(ALL_COLUMNS)
    assert any(not case[1].final for case in switches)


def test_phazing_has_its_own_priority_code(cases: list[Case]) -> None:
    phazed = matching(
        cases,
        lambda c: (
            bool(c[1].features) and move_name(c[0], c[1].index, c[3]) in ("roar", "whirlwind")
        ),
    )
    for _, transition, _, _ in phazed:
        assert transition.features["a_phaze"] == 1
        assert transition.features["a_priority"] == -1
        assert transition.features["a_prio_phaze"] == -2
        assert transition.features["a_status_mv"] == 1


def test_explosion_is_seen_by_the_damage_columns(cases: list[Case]) -> None:
    hits = matching(
        cases,
        lambda c: (
            bool(c[1].features)
            and move_name(c[0], c[1].index, c[3]) in ("explosion", "selfdestruct")
        ),
    )
    for _, transition, _, _ in hits:
        features = transition.features
        assert features["a_bp"] in (200, 250)
        assert features["a_status_mv"] == 0
        assert features["a_dmg"] > 0
        assert features["x_hp_minus_dmg"] == features["x_hp"] - features["a_dmg"]
        assert features["a_drop_any"] > 0


def test_a_described_transition_has_exactly_the_53_columns(cases: list[Case]) -> None:
    assert len(ALL_COLUMNS) == 53
    for _, transition, _, _ in matching(cases, lambda c: bool(c[1].features)):
        assert set(transition.features) == set(ALL_COLUMNS)


def test_the_label_does_not_change_the_description(
    records: list[Record], extractor: TransitionExtractor
) -> None:
    labelled = next(r for r in records if "player_won" in r)
    flipped = {**labelled, "player_won": 1 - labelled["player_won"]}
    described = repr(extractor.extract(without_label(labelled), "p1"))
    assert repr(extractor.extract(labelled, "p1")) == described
    assert repr(extractor.extract(flipped, "p1")) == described
