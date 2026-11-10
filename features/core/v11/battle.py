"""v11's view of one replayed battle, as its feature groups read it.

The view is built from a :class:`domain.battle.BattleContext`: the record's observed timeline
(``domain.timeline.ObservedTimeline``: both players entry by entry, events, revealed moves,
Spikes, levels and items), the battle's species (``context.species``) and the P2 faint
probabilities of the context's repair, whatever it is (``none``, ``rule``, ``model=…``,
``v11=…``): the groups only read ``p2_faint``, and the tracks take the larger of a probability
and a recorded faint. Player 2's lead counts as revealed even when it switches out on turn 1 and
never shows up in the timeline again.

The view only depends on the battle and its repair, so it is remembered on the context
(:func:`battle_of`), as cross-validation asks for the same battle in every fold.
"""

from __future__ import annotations

import math
from typing import cast

from domain.battle import BattleContext
from domain.pokedex import PokemonGen2
from domain.timeline import OPPONENT, ObservedTimeline

SIDES = ("p1", "p2")
OPP = OPPONENT
STATUSES = ("slp", "par", "frz", "tox", "psn", "brn")
VOLATILE = (
    "substitute",
    "confusion",
    "leechseed",
    "perish",
    "trapped",
    "encore",
    "attract",
    "destinybond",
    "protect",
    "curse",
    "nightmare",
)
NAN = float("nan")


class Battle:
    """Both sides of one battle, its Spikes, species and the repair's P2 faint probabilities."""

    def __init__(self, context: BattleContext) -> None:
        record = context.record
        self.id = record["battle_id"]
        self.team = record["p1_team_details"]
        self.lead2 = record["p2_lead_details"]
        self.species = context.species
        self.timeline = ObservedTimeline(record)
        self.n = self.timeline.n
        self.sides = self.timeline.sides
        self.spikes = self.timeline.spikes
        self.p2_faint = dict(context.repair.probabilities(record))

    def stats(self, name: str) -> PokemonGen2:
        """A species whose stats the record carries (P1's team, P2's lead)."""
        return cast(PokemonGen2, self.species(name))

    def level(self, side: str, name: str) -> int:
        """Level of ``name`` on ``side`` (100 when unknown)."""
        return self.timeline.level(side, name)

    def item(self, side: str, name: str) -> str | None:
        """Item of a P1 Pokémon; P2's items are unknown."""
        return self.timeline.item(side, name)


def battle_of(context: BattleContext) -> Battle:
    """The view of ``context``, built once per replay."""
    return context.cached(("v11", "battle"), lambda: Battle(context))


def log_ratio(a: float, b: float) -> float:
    """``log(a / b)``, NaN unless both are positive numbers."""
    if not (a > 0 and b > 0) or math.isnan(a) or math.isnan(b):
        return NAN
    return math.log(a / b)
