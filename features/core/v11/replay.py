"""v11's view of one battle: the observed timeline of both players, its Spikes, the species and
the P2 faint probabilities set by the hidden-faint detector.

The timeline, events, revealed moves and Spikes are ``domain.timeline.ObservedTimeline``; the
transitions of the detector are ``domain.switches.SwitchExtractor`` rows. Player 2's lead counts
as revealed even when it switches out on turn 1 and never shows up in the timeline again.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, cast

from domain.pokedex import PokemonGen2
from domain.switches import SwitchExtractor
from domain.timeline import OPPONENT, ObservedTimeline, SideTimeline
from features.core.v11.mechanics import Pokedex, Record
from features.core.v11.moves import CHART

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

Row = dict[str, Any]
Side = SideTimeline


class Battle:
    """Both sides of one battle, its Spikes and the P2 faint probabilities set by the detector."""

    def __init__(self, raw: Record, dex: Pokedex, cut: int | None = None) -> None:
        self.id = raw["battle_id"]
        self.team = raw["p1_team_details"]
        self.lead2 = raw["p2_lead_details"]
        self.species = dex.for_battle(raw).get
        self.record = {**raw, "battle_timeline": raw["battle_timeline"][:cut]}
        self.timeline = ObservedTimeline(self.record)
        self.n = self.timeline.n
        self.sides = self.timeline.sides
        self.spikes = self.timeline.spikes
        self.p2_faint: dict[int, float] = {}

    def stats(self, name: str) -> PokemonGen2:
        """A species whose stats the record carries (P1's team, P2's lead)."""
        return cast(PokemonGen2, self.species(name))

    def level(self, side: str, name: str) -> int:
        """Level of ``name`` on ``side`` (100 when unknown)."""
        return self.timeline.level(side, name)

    def item(self, side: str, name: str) -> str | None:
        """Item of a P1 Pokémon; P2's items are unknown."""
        return self.timeline.item(side, name)

    def transitions(self, s: str) -> list[Row]:
        """Active-Pokémon changes of side s, as rows (see ``domain.switches``)."""
        extractor = SwitchExtractor(self.species, CHART)
        return [
            {
                "i": t.index,
                "x": t.leaver,
                "recorded": t.recorded,
                "final": t.final,
                "label": t.label,
                **t.features,
            }
            for t in extractor.extract(self.record, "p1" if s == "p1" else "p2")
        ]


def parse(raws: Iterable[Record], dex: Pokedex, cut: int | None = None) -> list[Battle]:
    """One ``Battle`` per record, timelines truncated to ``cut`` entries when given."""
    return [Battle(r, dex, cut) for r in raws]


def log_ratio(a: float, b: float) -> float:
    """``log(a / b)``, NaN unless both are positive numbers."""
    if not (a > 0 and b > 0) or math.isnan(a) or math.isnan(b):
        return NAN
    return math.log(a / b)
