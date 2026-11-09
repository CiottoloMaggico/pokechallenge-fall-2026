"""v11 mechanics (hand-checked stats and damage) and replay (events, Spikes, transitions)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from features.core.v11.mechanics import Entry, Pokedex, damage_fraction, stat_value
from features.core.v11.replay import Battle, parse

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "battles.jsonl"

Record = dict[str, Any]


@pytest.fixture(scope="module")
def records() -> list[Record]:
    return [json.loads(line) for line in FIXTURES.open()]


@pytest.fixture(scope="module")
def battles(records: list[Record]) -> list[Battle]:
    return parse(records, Pokedex(records))


def entry(types: tuple[str, ...], base: int = 100) -> Entry:
    return {"hp": base, "atk": base, "def": base, "spa": base, "spd": base, "types": types}


def move(name: str, kind: str, category: str, power: int, accuracy: float = 1.0) -> Record:
    return {
        "name": name,
        "type": kind,
        "category": category,
        "base_power": power,
        "accuracy": accuracy,
    }


NORMAL, GHOST = entry(("normal",)), entry(("ghost",))
# Base 100 at level 100: 298 for a stat, 403 for HP; (2 * 100 / 5 + 2) = 42.
STAT, HP = 298, 403


def test_stat_value_by_hand() -> None:
    assert stat_value(100) == STAT
    assert stat_value(100, hp=True) == HP
    assert stat_value(160, hp=True) == 523
    assert stat_value(100, level=50) == (293 * 50) // 100 + 5


def test_damage_with_stab() -> None:
    hit = move("bodyslam", "normal", "physical", 100)
    expected = (42 * 100 * STAT / STAT / 50 + 2) * 1.5 * 0.925 / HP
    assert damage_fraction(hit, NORMAL, NORMAL) == pytest.approx(expected)
    no_stab = (42 * 100 / 50 + 2) * 0.925 / HP
    assert damage_fraction(hit, entry(("water",)), NORMAL) == pytest.approx(no_stab)


def test_damage_immunity_and_status_moves() -> None:
    assert damage_fraction(move("bodyslam", "normal", "physical", 100), NORMAL, GHOST) == 0.0
    assert damage_fraction(move("toxic", "poison", "status", 0), NORMAL, NORMAL) == 0.0
    assert damage_fraction(None, NORMAL, NORMAL) == 0.0


def test_seismic_toss_deals_the_level() -> None:
    toss = move("seismictoss", "fighting", "physical", 1)
    assert damage_fraction(toss, NORMAL, NORMAL, att_level=88) == pytest.approx(88 / HP)
    assert damage_fraction(toss, NORMAL, GHOST) == 0.0


def test_explosion_halves_defense() -> None:
    boom = move("explosion", "normal", "physical", 250)
    expected = (42 * 250 * STAT / (STAT * 0.5) / 50 + 2) * 1.5 * 0.925 / HP
    assert damage_fraction(boom, NORMAL, NORMAL) == pytest.approx(expected)


def test_burn_halves_physical_attack_only() -> None:
    hit = move("bodyslam", "normal", "physical", 100)
    expected = (42 * 100 * 0.5 / 50 + 2) * 1.5 * 0.925 / HP
    assert damage_fraction(hit, NORMAL, NORMAL, att_status="brn") == pytest.approx(expected)
    beam = move("triattack", "normal", "special", 80)
    assert damage_fraction(beam, NORMAL, NORMAL, att_status="brn") == damage_fraction(
        beam, NORMAL, NORMAL
    )


def test_thick_club_doubles_marowak_attack() -> None:
    quake = move("earthquake", "ground", "physical", 100)
    ground = entry(("ground",))
    expected = (42 * 100 * 2 / 50 + 2) * 1.5 * 0.925 / HP
    club = damage_fraction(quake, ground, NORMAL, att_item="thickclub", att_name="marowak")
    assert club == pytest.approx(expected)
    other = damage_fraction(quake, ground, NORMAL, att_item="thickclub", att_name="golem")
    assert other == damage_fraction(quake, ground, NORMAL)


def test_a_repeated_move_is_one_event(battles: list[Battle]) -> None:
    repeats = 0
    for battle in battles:
        for side in battle.sides.values():
            for i in range(1, side.n):
                now, before = side.moves[i], side.moves[i - 1]
                same = side.names[i] == side.names[i - 1]
                if same and now is not None and before is not None:
                    fresh = now["name"] != before["name"]
                    assert (side.events[i] is not None) == fresh
                    repeats += not fresh
    assert repeats > 0, "the fixtures must cover a repeated move"


def test_spikes_laid_and_spun(battles: list[Battle]) -> None:
    laid = spun = 0
    for battle in battles:
        for own, opp in (("p1", "p2"), ("p2", "p1")):
            spikes = battle.spikes[own]
            for i in range(1, battle.n):
                set_now = battle.sides[opp].events[i]
                spin_now = battle.sides[own].events[i]
                if spin_now is not None and spin_now["name"] == "rapidspin":
                    assert not spikes[i]
                    spun += spikes[i - 1]
                elif set_now is not None and set_now["name"] == "spikes":
                    assert spikes[i]
                    laid += not spikes[i - 1]
                else:
                    assert spikes[i] == spikes[i - 1]
    assert laid > 0 and spun > 0, "the fixtures must lay and spin Spikes"


def test_a_p1_faint_skips_the_fnt_row(battles: list[Battle]) -> None:
    found = 0
    for battle in battles:
        own, opp = battle.sides["p1"], battle.sides["p2"]
        for row in battle.transitions("p1"):
            i = row["i"]
            if own.status[i - 1] != "fnt":
                assert not row["recorded"] and row["label"] == 0
                continue
            assert row["recorded"] and row["label"] == 1
            usable = opp.status[i - 1] != "fnt" and i >= 2 and own.names[i - 2] == row["x"]
            assert ("x_hp" in row) == usable
            if usable:
                assert row["x_hp"] == own.hp[i - 2] and row["turn"] == i
                found += 1
    assert found > 0, "the fixtures must cover a P1 faint"


def test_p2_faints_are_never_usable(battles: list[Battle]) -> None:
    for battle in battles:
        for row in battle.transitions("p2"):
            assert row["recorded"] == ("x_hp" not in row)
