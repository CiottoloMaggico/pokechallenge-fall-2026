"""Hand-checked Gen 2 damage cases.

Every Pokémon has base 50 everywhere, level 100, max DV and stat experience:
each stat is ((50 + 15) * 2 + 63) * 100 // 100 + 5 = 198. A neutral 100-power
move then gives ((2 * 100 // 5 + 2) * 100 * 198 // 198) // 50 = 84, + 2 = 86,
with rolls from 86 * 217 // 255 = 73 to 86.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from domain.damage import Combatant, DamageCalculator, DamageRoll, Field
from domain.moves import CRIT_RATE, Move
from domain.pokedex import PokemonGen2
from domain.typechart import TypeChart

CHART = Path(__file__).resolve().parents[2] / "assets" / "type_chart_gen2.csv"


def species(*types: str, name: str = "test") -> PokemonGen2:
    return PokemonGen2(name, types, 50, 50, 50, 50, 50, 50)


def move(
    type_: str = "normal",
    category: str = "physical",
    power: int = 100,
    name: str = "testmove",
    accuracy: float = 1.0,
) -> Move:
    return Move(name, type_, category, power, accuracy, 0)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def damage() -> DamageCalculator:
    return DamageCalculator(TypeChart.from_csv(CHART))


WATER = Combatant(species("water"))

ROLL_CASES = {
    "neutral": (WATER, WATER, move(), Field(), False, 73, 86),
    # 86 * 1.5 = 129, 129 * 217 // 255 = 109
    "stab": (Combatant(species("normal")), WATER, move(), Field(), False, 109, 129),
    # 86 * 2 = 172, 172 * 217 // 255 = 146
    "super_effective": (WATER, WATER, move("electric", "special"), Field(), False, 146, 172),
    # 86 * 0.5 = 43, 43 * 217 // 255 = 36
    "resisted": (WATER, WATER, move("fire", "special"), Field(), False, 36, 43),
    # 86 * 2 * 2 = 344, 344 * 217 // 255 = 292
    "double_weakness": (
        WATER,
        Combatant(species("dragon", "flying")),
        move("ice", "special"),
        Field(),
        False,
        292,
        344,
    ),
    "immunity": (WATER, Combatant(species("ghost")), move(), Field(), False, 0, 0),
    # (84 * 2) + 2 = 170, 170 * 217 // 255 = 144
    "crit": (WATER, WATER, move(), Field(), True, 144, 170),
    # A 396, D 198 -> 99, 49; 42 * 100 * 99 // 49 // 50 = 169; 169 * 2 + 2 = 340
    "crit_keeps_favourable_stage": (
        Combatant(species("water"), boosts={"atk": 2}),
        WATER,
        move(),
        Field(),
        True,
        289,
        340,
    ),
    # stages ignored because attacker stage <= defender stage
    "crit_ignores_defense_stage": (
        WATER,
        Combatant(species("water"), boosts={"def": 2}),
        move(),
        Field(),
        True,
        144,
        170,
    ),
    # D 396 -> A 49, D 99; 42 * 100 * 49 // 99 // 50 = 41; + 2 = 43
    "reflect": (WATER, WATER, move(), Field(screens=frozenset({"reflect"})), False, 36, 43),
    "reflect_not_on_special": (
        WATER,
        WATER,
        move(category="special"),
        Field(screens=frozenset({"reflect"})),
        False,
        73,
        86,
    ),
    "reflect_ignored_by_crit": (
        WATER,
        WATER,
        move(),
        Field(screens=frozenset({"reflect"})),
        True,
        144,
        170,
    ),
    "light_screen": (
        WATER,
        WATER,
        move(category="special"),
        Field(screens=frozenset({"lightscreen"})),
        False,
        36,
        43,
    ),
    "light_screen_not_on_physical": (
        WATER,
        WATER,
        move(),
        Field(screens=frozenset({"lightscreen"})),
        False,
        73,
        86,
    ),
    # sandstorm hurts at the end of the turn, it does not change move damage
    "sand": (WATER, WATER, move("rock"), Field("sand"), False, 73, 86),
    # A 198 // 2 = 99; 42 * 100 * 99 // 198 // 50 = 42; + 2 = 44
    "burn": (Combatant(species("water"), status="brn"), WATER, move(), Field(), False, 37, 44),
    "burn_not_on_special": (
        Combatant(species("water"), status="brn"),
        WATER,
        move(category="special"),
        Field(),
        False,
        73,
        86,
    ),
    # D 198 // 2 = 99; 42 * 250 * 198 // 99 // 50 = 420; + 2 = 422
    "explosion": (WATER, WATER, move(power=250, name="explosion"), Field(), False, 359, 422),
    # recorded with power 0: real power 70, no type; 42 * 70 // 50 = 58; + 2 = 60
    "hidden_power": (
        WATER,
        WATER,
        move(category="special", power=0, name="hiddenpower"),
        Field(),
        False,
        51,
        60,
    ),
}


@pytest.mark.parametrize("case", ROLL_CASES)
def test_roll_table(damage: DamageCalculator, case: str) -> None:
    attacker, defender, used, field, crit, low, high = ROLL_CASES[case]
    result = damage.roll(attacker, defender, used, field, crit)
    assert result is not None
    assert (result.min, result.max) == (low, high)


@pytest.mark.parametrize(
    ("weather", "type_", "low", "high"),
    [
        # 86 * 1.5 = 129, 129 * 217 // 255 = 109
        ("rain", "water", 109, 129),
        # 86 * 0.5 = 43, 43 * 217 // 255 = 36
        ("rain", "fire", 36, 43),
        ("sun", "fire", 109, 129),
        ("sun", "water", 36, 43),
        ("rain", "normal", 73, 86),
    ],
)
def test_rain_and_sun(
    damage: DamageCalculator, weather: str, type_: str, low: int, high: int
) -> None:
    attacker = Combatant(species("electric"))
    defender = Combatant(species("normal"))
    result = damage.roll(attacker, defender, move(type_, "special"), Field.of(weather))
    assert result is not None
    assert (result.min, result.max) == (low, high)


def test_mean_is_the_average_of_the_39_rolls(damage: DamageCalculator) -> None:
    result = damage.roll(WATER, WATER, move())
    assert result is not None
    assert result.mean == pytest.approx(sum(86 * r // 255 for r in range(217, 256)) / 39)


def test_multi_hit_counts_the_expected_hits(damage: DamageCalculator) -> None:
    # power 100 as above, times the 3 expected hits of a 2-5 hit move
    result = damage.roll(WATER, WATER, move(name="furyattack"))
    assert result is not None
    assert (result.min, result.max) == (73 * 3, 86 * 3)


@pytest.mark.parametrize(
    ("used", "expected"),
    [
        (move("dragon", "special", 0, "dragonrage"), 40),
        (move("normal", "special", 0, "sonicboom"), 20),
        (move("fighting", "physical", 0, "seismictoss"), 100),
        (move("psychic", "special", 0, "psywave"), 75),
        # max HP (50 + 15) * 2 + 63 + 100 + 10 = 303, half of it rounded down
        (move("normal", "physical", 0, "superfang"), 151),
        (move("normal", "physical", 0, "guillotine"), 303),
    ],
)
def test_moves_outside_the_formula(damage: DamageCalculator, used: Move, expected: float) -> None:
    assert damage.roll(WATER, WATER, used) == DamageRoll.constant(expected)


def test_ohko_fails_on_a_higher_level(damage: DamageCalculator) -> None:
    weak = Combatant(species("water"), level=50)
    guillotine = move("normal", "physical", 0, "guillotine")
    assert damage.roll(weak, WATER, guillotine) == DamageRoll.constant(0)


@pytest.mark.parametrize(
    "used",
    [
        move(category="status", power=0, name="toxic"),
        move("fighting", "physical", 0, "counter"),
    ],
)
def test_no_prediction(damage: DamageCalculator, used: Move) -> None:
    assert damage.roll(WATER, WATER, used) is None
    assert damage.expected(WATER, WATER, used, with_accuracy=True) is None


def test_expected_mixes_crits_and_applies_accuracy_on_request(damage: DamageCalculator) -> None:
    used = move(accuracy=0.5)
    normal = damage.roll(WATER, WATER, used)
    critical = damage.roll(WATER, WATER, used, crit=True)
    assert normal is not None and critical is not None
    mixed = (1 - CRIT_RATE) * normal.mean + CRIT_RATE * critical.mean
    assert damage.expected(WATER, WATER, used, with_accuracy=False) == pytest.approx(mixed)
    assert damage.expected(WATER, WATER, used, with_accuracy=True) == pytest.approx(mixed * 0.5)


def test_expected_requires_with_accuracy_by_keyword(damage: DamageCalculator) -> None:
    with pytest.raises(TypeError):
        damage.expected(WATER, WATER, move())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        damage.expected(WATER, WATER, move(), Field(), True)  # type: ignore[misc]


def test_hits_to_ko_ignores_accuracy(damage: DamageCalculator) -> None:
    # 303 HP / about 85 per use, whatever the accuracy
    assert damage.hits_to_ko(WATER, WATER, move(accuracy=0.5)) == 4
    assert damage.hits_to_ko(WATER, Combatant(species("ghost")), move()) == math.inf


def test_paralysis_divides_speed_only() -> None:
    paralysed = Combatant(species("water"), status="par")
    assert paralysed.stat("spe") == 198 // 4
    assert paralysed.stat("atk") == 198


def test_field_of_collects_the_screens() -> None:
    assert Field.of("rain", ["reflect"]) == Field("rain", frozenset({"reflect"}))
    assert Field() == Field.of(None)


def test_fraction_of_scales_the_three_values() -> None:
    assert DamageRoll(10, 20, 15).fraction_of(10) == DamageRoll(1.0, 2.0, 1.5)
