"""A quick, roll-free estimate of the damage of one move, as a share of the defender's maximal HP.

:class:`domain.damage.DamageCalculator` applies the Gen 2 formula with its rolls, critical hits,
items and field; this estimate keeps only what decides most matchups, so it is cheap enough to
fill whole matchup matrices. Its assumptions (each a default of the functions below):

* stats with maximal DVs (15) and maximal stat experience (63 points) at the Pokémon's level,
  boost stages applied as ``(2 + s) / 2`` or ``2 / (2 - s)``;
* the mean damage roll (factor 0.925) weighted by the move's accuracy;
* STAB 1.5; a burn halves the physical Attack; Thick Club (Cubone, Marowak) doubles the
  physical Attack and Light Ball (Pikachu) the Special Attack of the attacker holding it;
  Explosion and Self-Destruct halve the defence;
* Seismic Toss and Night Shade deal the attacker's level, other moves of base power 0 or 1
  (variable power) and status moves nothing;
* type effectiveness over a :class:`domain.typechart.TypeChart`, where the placeholder type
  ``notype`` is skipped and a type outside ``domain.pokedex.TYPES`` (the records' ``???`` of
  Curse) counts as neutral.

Species are :class:`domain.pokedex.PokemonGen2` (None for a species without stats: no damage,
NaN speed); moves are the records' move details (mappings with ``name``, ``type``,
``category``, ``base_power``, ``accuracy``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from domain.pokedex import PLACEHOLDER_TYPES, TYPES, PokemonGen2
from domain.typechart import TypeChart

Move = Mapping[str, Any]

MAX_DV = 15
STAT_POINTS = 63
MEAN_ROLL = 0.925
STAB = 1.5
FIXED_DAMAGE = frozenset({"seismictoss", "nightshade"})
HALVES_DEFENCE = frozenset({"explosion", "selfdestruct"})
THICK_CLUB_USERS = ("marowak", "cubone")
LIGHT_BALL_USERS = ("pikachu",)
PARALYSIS_SPEED = 0.25
NEUTRAL = 1.0


def stat_value(base: float, level: int = 100, hp: bool = False) -> float:
    """Gen 2 stat with maximal DVs and stat experience (an int for an int base)."""
    core = ((base + MAX_DV) * 2 + STAT_POINTS) * level // 100
    return core + level + 10 if hp else core + 5


def stage_mult(stage: int) -> float:
    """Multiplier of a boost stage."""
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def effectiveness(chart: TypeChart, move_type: str, defender_types: Iterable[str]) -> float:
    """Product of the multipliers over the defender's types (see the module for the rules)."""
    mult = NEUTRAL
    for defending in defender_types:
        if defending in PLACEHOLDER_TYPES:
            continue
        known = move_type in TYPES and defending in TYPES
        mult *= chart.multiplier(move_type, defending) if known else NEUTRAL
    return mult


def stab_effectiveness(
    chart: TypeChart, attacker_types: Sequence[str], defender_types: Sequence[str]
) -> float:
    """Best multiplier of the attacker's own types (1.0 without types)."""
    return max((effectiveness(chart, t, defender_types) for t in attacker_types), default=NEUTRAL)


def expected_damage(
    chart: TypeChart,
    move: Move | None,
    att: PokemonGen2 | None,
    dfn: PokemonGen2 | None,
    att_level: int = 100,
    def_level: int = 100,
    att_boosts: Mapping[str, int] | None = None,
    def_boosts: Mapping[str, int] | None = None,
    att_item: str | None = None,
    att_status: str | None = None,
    att_name: str | None = None,
) -> float:
    """Expected share of the defender's maximal HP taken by ``move`` (accuracy included)."""
    if move is None or move["category"] == "status" or att is None or dfn is None:
        return 0.0
    def_hp = stat_value(dfn.base_hp, def_level, hp=True)
    eff = effectiveness(chart, move["type"], dfn.types)
    if move["name"] in FIXED_DAMAGE:
        return 0.0 if eff == 0 else att_level / def_hp
    power = move["base_power"]
    if power <= 1:
        return 0.0
    physical = move["category"] == "physical"
    a_key, d_key = ("atk", "def") if physical else ("spa", "spd")
    a = stat_value(att.base(a_key), att_level) * stage_mult((att_boosts or {}).get(a_key, 0))
    d = stat_value(dfn.base(d_key), def_level) * stage_mult((def_boosts or {}).get(d_key, 0))
    if physical and att_status == "brn":
        a *= 0.5
    if att_item == "thickclub" and att_name in THICK_CLUB_USERS and physical:
        a *= 2
    if att_item == "lightball" and att_name in LIGHT_BALL_USERS and not physical:
        a *= 2
    if move["name"] in HALVES_DEFENCE:
        d *= 0.5
    base = (2 * att_level / 5 + 2) * power * a / d / 50 + 2
    stab = STAB if move["type"] in att.types else 1.0
    return base * stab * eff * MEAN_ROLL * move["accuracy"] / def_hp


def best_damage(
    chart: TypeChart,
    moves: Iterable[Move | None],
    att: PokemonGen2 | None,
    dfn: PokemonGen2 | None,
    att_level: int = 100,
    def_level: int = 100,
    att_boosts: Mapping[str, int] | None = None,
    def_boosts: Mapping[str, int] | None = None,
    att_item: str | None = None,
    att_status: str | None = None,
    att_name: str | None = None,
) -> float:
    """The largest :func:`expected_damage` over ``moves`` (0 without moves)."""
    best = 0.0
    for move in moves:
        damage = expected_damage(
            chart, move, att, dfn, att_level, def_level, att_boosts, def_boosts,
            att_item, att_status, att_name,
        )  # fmt: skip
        best = max(best, damage)
    return best


def effective_speed(
    species: PokemonGen2 | None, boosts: Mapping[str, int], status: str | None, level: int = 100
) -> float:
    """Speed after boosts and paralysis (NaN for a species without stats)."""
    if species is None:
        return float("nan")
    value = stat_value(species.base_spe, level) * stage_mult(boosts.get("spe", 0))
    return value * PARALYSIS_SPEED if status == "par" else value
