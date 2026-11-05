"""Gen 2 moves: the data seen in battle records and the damage rules.

A :class:`Move` is what the records contain (name, type, category, base
power, accuracy, priority). Several moves do not deal damage through the
standard formula, and some arrive with placeholder data (Hidden Power is
recorded as a 0-power Normal move, Return and Frustration with power 0).
:func:`damage_kind`, :func:`power`, :func:`attack_type` and :func:`hits`
describe how much damage a move really deals, for ``damage.py``.

Variable outcomes are replaced by their expected value (Magnitude,
Present, multi-hit moves, Psywave); unknowable ones (Counter, Mirror
Coat, Bide, Beat Up) are marked as :attr:`DamageKind.UNPREDICTABLE`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from domain.records import CATEGORIES, MoveCategory


class DamageKind(Enum):
    """How a move's damage is determined."""

    NONE = "none"  # status move
    STANDARD = "standard"  # Gen 2 damage formula
    FIXED = "fixed"  # constant HP (Dragon Rage, Sonic Boom)
    LEVEL = "level"  # proportional to the user's level (Seismic Toss)
    HALF_HP = "half_hp"  # half of the target's current HP (Super Fang)
    OHKO = "ohko"  # knocks out if it hits
    UNPREDICTABLE = "unpredictable"  # depends on damage taken or the party


FIXED_DAMAGE = {"dragonrage": 40, "sonicboom": 20}
LEVEL_DAMAGE = {"seismictoss": 1.0, "nightshade": 1.0, "psywave": 0.75}
HALF_HP_DAMAGE = frozenset({"superfang"})
OHKO_MOVES = frozenset({"fissure", "guillotine", "horndrill"})
UNPREDICTABLE_MOVES = frozenset({"counter", "mirrorcoat", "bide", "beatup"})

# real (or expected) power of moves recorded with a placeholder power
VARIABLE_POWER = {
    "hiddenpower": 70,  # maximum; its type depends on hidden DVs
    "return": 102,  # maximum happiness
    "frustration": 102,  # minimum happiness
    "magnitude": 71,  # expected power over magnitudes 4..10
    "present": 52,  # expected power, healing outcome counted as 0
}
TYPELESS_MOVES = frozenset({"hiddenpower", "struggle"})
HP_BASED_POWER = frozenset({"flail", "reversal"})

# expected number of hits (2-5 hit moves: 3/8, 3/8, 1/8, 1/8 -> 3.0)
MULTI_HIT = {
    "doublekick": 2.0,
    "twineedle": 2.0,
    "bonemerang": 2.0,
    "triplekick": 6.0,  # 10 + 20 + 30 power, expressed as hits of 10
    **{
        name: 3.0
        for name in (
            "barrage",
            "bonerush",
            "cometpunch",
            "doubleslap",
            "furyattack",
            "furyswipes",
            "pinmissile",
            "spikecannon",
        )
    },
}
SELF_KO_MOVES = frozenset({"explosion", "selfdestruct"})  # halve target Defense
HIGH_CRIT_MOVES = frozenset(
    {"aeroblast", "crabhammer", "crosschop", "karatechop", "razorleaf", "slash"}
)
CRIT_RATE = 17 / 256
HIGH_CRIT_RATE = 64 / 256

# Flail / Reversal: (minimum of 48 * hp / max_hp, power), first match wins
_HP_POWER_TABLE = ((33, 20), (17, 40), (10, 80), (5, 100), (2, 150), (0, 200))


@dataclass(frozen=True)
class Move:
    """A move as recorded in the battle timeline."""

    name: str
    type: str
    category: MoveCategory
    base_power: int
    accuracy: float
    priority: int

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.name}: unknown category {self.category!r}")
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValueError(f"{self.name}: accuracy must be in [0, 1]")

    @classmethod
    def from_details(cls, details: Mapping[str, Any]) -> Move:
        """Build a move from a record's ``MoveDetails``."""
        return cls(
            details["name"],
            details["type"],
            details["category"],
            int(details["base_power"]),
            float(details["accuracy"]),
            int(details["priority"]),
        )

    @property
    def is_status(self) -> bool:
        """Whether the move deals no direct damage."""
        return self.category == "status"

    @property
    def is_physical(self) -> bool:
        """Whether the move uses Attack and Defense."""
        return self.category == "physical"


def damage_kind(move: Move) -> DamageKind:
    """How the damage of ``move`` is determined."""
    name = move.name
    if move.is_status:
        return DamageKind.NONE
    if name in UNPREDICTABLE_MOVES:
        return DamageKind.UNPREDICTABLE
    if name in FIXED_DAMAGE:
        return DamageKind.FIXED
    if name in LEVEL_DAMAGE:
        return DamageKind.LEVEL
    if name in HALF_HP_DAMAGE:
        return DamageKind.HALF_HP
    if name in OHKO_MOVES:
        return DamageKind.OHKO
    return DamageKind.STANDARD


def power(move: Move, user_hp_fraction: float = 1.0) -> float:
    """Power used in the damage formula (for STANDARD moves, per hit).

    ``user_hp_fraction`` matters only for Flail and Reversal.
    """
    if move.name in HP_BASED_POWER:
        ratio = int(48 * max(0.0, min(1.0, user_hp_fraction)))
        return next(p for threshold, p in _HP_POWER_TABLE if ratio >= threshold)
    if move.name in VARIABLE_POWER:
        return VARIABLE_POWER[move.name]
    if move.name == "triplekick":
        return 10
    return move.base_power


def attack_type(move: Move) -> str | None:
    """Type used for STAB and effectiveness; None for typeless moves."""
    return None if move.name in TYPELESS_MOVES else move.type


def hits(move: Move) -> float:
    """Expected number of hits (1 for single-hit moves)."""
    return MULTI_HIT.get(move.name, 1.0)


def crit_rate(move: Move) -> float:
    """Chance of a critical hit without Focus Energy or items."""
    return HIGH_CRIT_RATE if move.name in HIGH_CRIT_MOVES else CRIT_RATE
