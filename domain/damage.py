"""Gen 2 damage calculation.

The formula follows the Gen 2 games::

    base = ((2 * L // 5 + 2) * power * A // D) // 50
    base = base * item boost, * 2 on a critical hit, capped at 997, + 2
    base = base * weather * STAB * effectiveness of each defending type
    damage = base * r // 255 for r in 217..255 (at least 1)

with A and D the boosted attacking and defending stats (both divided by
4 when one exceeds 255), Burn halving physical Attack, Reflect / Light
Screen doubling Defense and Explosion halving the target's Defense.

Example::

    chart = TypeChart.from_csv("assets/type_chart_gen2.csv")
    snorlax = Combatant(pokedex.get("snorlax"))
    zapdos = Combatant(pokedex.get("zapdos"))
    result = calc_damage(zapdos, snorlax, thunderbolt, chart)
    result.min, result.max, result.fraction_of(snorlax.max_hp)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from domain.moves import (
    FIXED_DAMAGE,
    LEVEL_DAMAGE,
    SELF_KO_MOVES,
    DamageKind,
    Move,
    attack_type,
    crit_rate,
    damage_kind,
    hits,
    power,
)
from domain.pokedex import DEFAULT_LEVEL, PokemonGen2
from domain.typechart import TypeChart

MIN_ROLL, MAX_ROLL = 217, 255
DAMAGE_CAP = 997
STAT_CAP = 999
STAB = 1.5
PARALYSIS_SPEED = 4  # Gen 2: a paralysed Pokémon's Speed is divided by 4

# Gen 2 stat stage multipliers, as fractions of 100
STAGE_MULTIPLIER = {
    -6: 25, -5: 28, -4: 33, -3: 40, -2: 50, -1: 66,
    0: 100, 1: 150, 2: 200, 3: 250, 4: 300, 5: 350, 6: 400,
}  # fmt: skip

TYPE_BOOST_ITEMS = {
    "blackbelt": "fighting",
    "blackglasses": "dark",
    "charcoal": "fire",
    "hardstone": "rock",
    "magnet": "electric",
    "metalcoat": "steel",
    "miracleseed": "grass",
    "mysticwater": "water",
    "nevermeltice": "ice",
    "pinkbow": "normal",
    "poisonbarb": "poison",
    "polkadotbow": "normal",
    "sharpbeak": "flying",
    "silverpowder": "bug",
    "softsand": "ground",
    "spelltag": "ghost",
    "twistedspoon": "psychic",
}
# item -> (holders, stat, multiplier as a fraction of 100)
SPECIES_ITEMS = {
    "thickclub": ({"cubone", "marowak"}, "atk", 200),
    "lightball": ({"pikachu"}, "spa", 200),
    "metalpowder": ({"ditto"}, "def", 150),
}
# item a species almost always holds, assumed when its item is unknown
# (P2's Pokémon): in the training teams 97% of Marowak hold a Thick Club
SIGNATURE_ITEMS = {
    "cubone": "thickclub",
    "marowak": "thickclub",
    "pikachu": "lightball",
    "ditto": "metalpowder",
}
WEATHER_MULTIPLIER = {
    ("rain", "water"): 1.5,
    ("rain", "fire"): 0.5,
    ("sun", "fire"): 1.5,
    ("sun", "water"): 0.5,
}


class PokemonLike(Protocol):
    """What :meth:`Combatant.from_state` reads (``battle.PokemonState``)."""

    @property
    def info(self) -> PokemonGen2: ...

    @property
    def level(self) -> int: ...

    @property
    def hp_pct(self) -> float: ...

    @property
    def status(self) -> str: ...

    @property
    def item(self) -> str | None: ...


@dataclass(frozen=True)
class Combatant:
    """A Pokémon as seen by the damage formula."""

    species: PokemonGen2
    level: int = DEFAULT_LEVEL
    hp_fraction: float = 1.0
    boosts: Mapping[str, int] = field(default_factory=dict)
    status: str = "nostatus"
    item: str | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.level <= 100:
            raise ValueError(f"{self.species.name}: level must be in 1..100")
        if not 0.0 <= self.hp_fraction <= 1.0:
            raise ValueError(f"{self.species.name}: hp_fraction must be in [0, 1]")
        if any(not -6 <= b <= 6 for b in self.boosts.values()):
            raise ValueError(f"{self.species.name}: boosts must be in -6..6")

    @classmethod
    def from_state(
        cls, pokemon: PokemonLike, boosts: Mapping[str, int] | None = None
    ) -> Combatant:
        """Build from a ``battle.PokemonState`` (and the active boosts)."""
        return cls(
            species=pokemon.info,
            level=pokemon.level,
            hp_fraction=pokemon.hp_pct,
            boosts=dict(boosts or {}),
            status=pokemon.status,
            item=pokemon.item,
        )

    @property
    def likely_item(self) -> str | None:
        """The species' signature item (Thick Club, Light Ball, ...)."""
        return SIGNATURE_ITEMS.get(self.species.name)

    @property
    def types(self) -> tuple[str, ...]:
        return self.species.types

    @property
    def max_hp(self) -> int:
        return self.species.max_hp(self.level)

    @property
    def hp(self) -> int:
        return math.ceil(self.hp_fraction * self.max_hp)

    def stat(self, name: str, boosted: bool = True) -> int:
        """Stat value with the held item and, optionally, the stat stage.

        Paralysis divides Speed by 4. An unknown item (None) is taken to
        be the species' signature item, if it has one.
        """
        value = self.species.stat(name, self.level)
        item = self.item if self.item is not None else self.likely_item
        holders, item_stat, factor = SPECIES_ITEMS.get(item or "", ((), "", 0))
        if item_stat == name and self.species.name in holders:
            value = value * factor // 100
        if boosted:
            value = value * STAGE_MULTIPLIER[self.boosts.get(name, 0)] // 100
        if name == "spe" and self.status == "par":
            value //= PARALYSIS_SPEED
        return max(1, min(STAT_CAP, value))


@dataclass(frozen=True)
class Field:
    """Conditions that change damage: weather and the defender's screens."""

    weather: str | None = None
    defender_screens: frozenset[str] = frozenset()

    @classmethod
    def of(cls, weather: str | None, defender_screens: Iterable[str] = ()) -> Field:
        return cls(weather, frozenset(defender_screens))


@dataclass(frozen=True)
class DamageRange:
    """Damage of one use of a move, over the 39 random rolls."""

    min: float
    max: float
    mean: float

    def fraction_of(self, hp: float) -> DamageRange:
        """The same range as a fraction of ``hp`` (e.g. the target's max HP)."""
        return DamageRange(self.min / hp, self.max / hp, self.mean / hp)

    @classmethod
    def constant(cls, value: float) -> DamageRange:
        return cls(value, value, value)


def calc_damage(
    attacker: Combatant,
    defender: Combatant,
    move: Move,
    chart: TypeChart,
    conditions: Field | None = None,
    crit: bool = False,
) -> DamageRange | None:
    """Damage of ``move`` from ``attacker`` to ``defender``.

    Returns None for status moves and moves whose damage cannot be
    predicted (Counter, Mirror Coat, Bide, Beat Up). Multi-hit moves
    return the damage of the expected number of hits.
    """
    kind = damage_kind(move)
    if kind in (DamageKind.NONE, DamageKind.UNPREDICTABLE):
        return None
    effectiveness = chart.effectiveness(attack_type(move), defender.types)
    if effectiveness == 0:
        return DamageRange.constant(0)
    if kind is DamageKind.FIXED:
        return DamageRange.constant(FIXED_DAMAGE[move.name])
    if kind is DamageKind.LEVEL:
        return DamageRange.constant(attacker.level * LEVEL_DAMAGE[move.name])
    if kind is DamageKind.HALF_HP:
        return DamageRange.constant(max(1, defender.hp // 2))
    if kind is DamageKind.OHKO:
        ko = attacker.level >= defender.level
        return DamageRange.constant(defender.hp if ko else 0)
    return _standard(attacker, defender, move, chart, conditions or Field(), crit)


def expected_damage(
    attacker: Combatant,
    defender: Combatant,
    move: Move,
    chart: TypeChart,
    conditions: Field | None = None,
    include_accuracy: bool = False,
) -> float | None:
    """Mean damage, mixing critical and normal hits by their chance.

    With ``include_accuracy`` the result is also multiplied by the
    move's accuracy (expected damage per attempt).
    """
    normal = calc_damage(attacker, defender, move, chart, conditions)
    if normal is None:
        return None
    mean = normal.mean
    if damage_kind(move) is DamageKind.STANDARD:
        critical = calc_damage(attacker, defender, move, chart, conditions, True)
        rate = crit_rate(move)
        mean = (1 - rate) * normal.mean + rate * (critical or normal).mean
    return mean * move.accuracy if include_accuracy else mean


def hits_to_ko(
    attacker: Combatant,
    defender: Combatant,
    move: Move,
    chart: TypeChart,
    conditions: Field | None = None,
) -> float | None:
    """Expected uses of ``move`` to knock out ``defender`` from its HP.

    ``math.inf`` if the move deals no damage, None if unpredictable.
    """
    mean = expected_damage(attacker, defender, move, chart, conditions)
    if mean is None:
        return None
    if mean <= 0:
        return math.inf
    return math.ceil(defender.hp / mean)


def _standard(
    attacker: Combatant,
    defender: Combatant,
    move: Move,
    chart: TypeChart,
    conditions: Field,
    crit: bool,
) -> DamageRange | None:
    move_power = power(move, attacker.hp_fraction)
    if move_power <= 0:
        return None
    attack, defense = _stats(attacker, defender, move, conditions, crit)
    base = ((2 * attacker.level // 5 + 2) * int(move_power) * attack // defense) // 50
    move_type = attack_type(move)
    if move_type is not None and TYPE_BOOST_ITEMS.get(attacker.item or "") == move_type:
        base = base * 110 // 100
    if crit:
        base *= 2
    base = min(base, DAMAGE_CAP) + 2
    if move_type is not None:
        if conditions.weather is not None:
            weather = WEATHER_MULTIPLIER.get((conditions.weather, move_type), 1.0)
            base = int(base * weather)
        if move_type in attacker.types:
            base = int(base * STAB)
        for defending_type in defender.types:
            base = int(base * chart.multiplier(move_type, defending_type))
    rolls = [max(1, base * r // 255) for r in range(MIN_ROLL, MAX_ROLL + 1)]
    n_hits = hits(move)
    return DamageRange(
        min(rolls) * n_hits, max(rolls) * n_hits, sum(rolls) / len(rolls) * n_hits
    )


def _stats(
    attacker: Combatant,
    defender: Combatant,
    move: Move,
    conditions: Field,
    crit: bool,
) -> tuple[int, int]:
    """Attacking and defending stat after stages, burn, screens and caps."""
    atk_name, def_name = ("atk", "def") if move.is_physical else ("spa", "spd")
    screen = "reflect" if move.is_physical else "lightscreen"
    # a critical hit ignores stages and screens unless that favours the user
    ignore = crit and attacker.boosts.get(atk_name, 0) <= defender.boosts.get(
        def_name, 0
    )
    attack = attacker.stat(atk_name, boosted=not ignore)
    defense = defender.stat(def_name, boosted=not ignore)
    if move.is_physical and attacker.status == "brn" and not ignore:
        attack //= 2
    if screen in conditions.defender_screens and not ignore:
        defense *= 2
    if move.name in SELF_KO_MOVES:
        defense //= 2
    if attack > 255 or defense > 255:
        attack, defense = attack // 4, defense // 4
    return max(1, attack), max(1, defense)
