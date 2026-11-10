"""The roll-free damage estimate, checked by hand on base-100 species."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from domain.expected_damage import (
    best_damage,
    effective_speed,
    effectiveness,
    expected_damage,
    stab_effectiveness,
    stage_mult,
    stat_value,
)
from domain.pokedex import PokemonGen2
from domain.typechart import TypeChart

ROOT = Path(__file__).resolve().parents[2]
CHART = TypeChart.from_csv(ROOT / "assets" / "type_chart_gen2.csv")
# Base 100 at level 100: 298 for a stat, 403 for HP; (2 * 100 / 5 + 2) = 42.
STAT, HP = 298, 403


def species(types: tuple[str, ...], base: int = 100, name: str = "x") -> PokemonGen2:
    return PokemonGen2(name, types, base, base, base, base, base, base)


def move(name: str, kind: str, category: str, power: int, accuracy: float = 1.0) -> dict[str, Any]:
    return {
        "name": name,
        "type": kind,
        "category": category,
        "base_power": power,
        "accuracy": accuracy,
    }


NORMAL, GHOST, WATER = species(("normal",)), species(("ghost",)), species(("water",))


def damage(m: dict[str, Any] | None, att: PokemonGen2, dfn: PokemonGen2, **kw: Any) -> float:
    return expected_damage(CHART, m, att, dfn, **kw)


def test_stat_value_by_hand() -> None:
    assert stat_value(100) == STAT
    assert stat_value(100, hp=True) == HP
    assert stat_value(160, hp=True) == 523
    assert stat_value(100, level=50) == (293 * 50) // 100 + 5
    assert isinstance(stat_value(100), int)


def test_stage_multipliers() -> None:
    assert [stage_mult(s) for s in (-2, -1, 0, 1, 2)] == [0.5, 2 / 3, 1.0, 1.5, 2.0]


def test_damage_with_and_without_stab() -> None:
    hit = move("bodyslam", "normal", "physical", 100)
    assert damage(hit, NORMAL, NORMAL) == pytest.approx((42 * 100 / 50 + 2) * 1.5 * 0.925 / HP)
    assert damage(hit, WATER, NORMAL) == pytest.approx((42 * 100 / 50 + 2) * 0.925 / HP)


def test_accuracy_weights_the_damage() -> None:
    full = damage(move("thunder", "electric", "special", 120), NORMAL, NORMAL)
    assert damage(move("thunder", "electric", "special", 120, 0.7), NORMAL, NORMAL) == (
        pytest.approx(0.7 * full)
    )


def test_immunity_status_moves_and_unknown_species_deal_nothing() -> None:
    assert damage(move("bodyslam", "normal", "physical", 100), NORMAL, GHOST) == 0.0
    assert damage(move("toxic", "poison", "status", 0), NORMAL, NORMAL) == 0.0
    assert damage(move("return", "normal", "physical", 1), NORMAL, NORMAL) == 0.0
    assert damage(None, NORMAL, NORMAL) == 0.0
    assert expected_damage(CHART, move("surf", "water", "special", 95), None, NORMAL) == 0.0


def test_seismic_toss_deals_the_level() -> None:
    toss = move("seismictoss", "fighting", "physical", 1)
    assert damage(toss, NORMAL, NORMAL, att_level=88) == pytest.approx(88 / HP)
    assert damage(toss, NORMAL, GHOST) == 0.0


def test_explosion_halves_the_defence() -> None:
    boom = move("explosion", "normal", "physical", 250)
    expected = (42 * 250 * STAT / (STAT * 0.5) / 50 + 2) * 1.5 * 0.925 / HP
    assert damage(boom, NORMAL, NORMAL) == pytest.approx(expected)


def test_burn_halves_the_physical_attack_only() -> None:
    hit = move("bodyslam", "normal", "physical", 100)
    expected = (42 * 100 * 0.5 / 50 + 2) * 1.5 * 0.925 / HP
    assert damage(hit, NORMAL, NORMAL, att_status="brn") == pytest.approx(expected)
    beam = move("triattack", "normal", "special", 80)
    assert damage(beam, NORMAL, NORMAL, att_status="brn") == damage(beam, NORMAL, NORMAL)


def test_thick_club_and_light_ball_double_their_holders_attack() -> None:
    quake = move("earthquake", "ground", "physical", 100)
    ground = species(("ground",))
    expected = (42 * 100 * 2 / 50 + 2) * 1.5 * 0.925 / HP
    club = damage(quake, ground, NORMAL, att_item="thickclub", att_name="marowak")
    assert club == pytest.approx(expected)
    other = damage(quake, ground, NORMAL, att_item="thickclub", att_name="golem")
    assert other == damage(quake, ground, NORMAL)
    bolt = move("thunderbolt", "electric", "special", 95)
    pikachu = species(("electric",), name="pikachu")
    ball = damage(bolt, pikachu, NORMAL, att_item="lightball", att_name="pikachu")
    assert ball > damage(bolt, pikachu, NORMAL)


def test_boosts_change_attack_and_defence() -> None:
    hit = move("bodyslam", "normal", "physical", 100)
    boosted = damage(hit, NORMAL, NORMAL, att_boosts={"atk": 2})
    assert boosted == pytest.approx((42 * 100 * 2 / 50 + 2) * 1.5 * 0.925 / HP)
    walled = damage(hit, NORMAL, NORMAL, def_boosts={"def": 2})
    assert walled == pytest.approx((42 * 100 / 2 / 50 + 2) * 1.5 * 0.925 / HP)


def test_effectiveness_skips_notype_and_unknown_types_are_neutral() -> None:
    assert effectiveness(CHART, "fire", ("grass", "notype")) == 2.0
    assert effectiveness(CHART, "ground", ("fire", "flying")) == 0.0
    assert effectiveness(CHART, "threequestionmarks", ("grass",)) == 1.0
    assert effectiveness(CHART, "fire", ()) == 1.0


def test_best_damage_and_stab_effectiveness() -> None:
    moves = [None, move("toxic", "poison", "status", 0), move("surf", "water", "special", 95)]
    assert best_damage(CHART, moves, NORMAL, NORMAL) == damage(moves[2], NORMAL, NORMAL)
    assert best_damage(CHART, [], NORMAL, NORMAL) == 0.0
    assert stab_effectiveness(CHART, ("fire", "water"), ("grass",)) == 2.0
    assert stab_effectiveness(CHART, (), ("grass",)) == 1.0


def test_effective_speed() -> None:
    assert effective_speed(NORMAL, {}, None) == STAT
    assert effective_speed(NORMAL, {"spe": 1}, "par") == STAT * 1.5 * 0.25
    assert math.isnan(effective_speed(None, {}, None))
