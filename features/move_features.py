"""How well each player's attacks matched the opposing Pokémon so far.

``mv_res``: share of P2's attacks that were resisted (effectiveness < 1
against the Pokémon hit) minus the same share for P1. Positive when P2
attacked into resistances more often, i.e. P1 is ahead. Only attacks
whose damage depends on type matchups count: fixed-damage moves
(Seismic Toss, Super Fang, Counter, ...) and Rapid Spin, used to clear
Spikes, are left out. Hidden Power and Struggle are typeless and count
as neutral.
"""

from __future__ import annotations

import math

from domain.battle import BattleContext
from domain.moves import DamageKind, Move, attack_type, damage_kind
from domain.typechart import TypeChart
from features.turn_features import AtTurn, opponent

UTILITY_ATTACKS = frozenset({"rapidspin"})  # used to clear Spikes, not to hurt


def resisted_share(chart: TypeChart, context: BattleContext, side: str) -> float:
    """Resisted attacks of ``side`` over its attacks (0 without attacks), NaN
    when the types of a target are unknown."""
    attacks = resisted = 0
    for turn in context.history:
        move = turn.side(side).move
        if move is None or not _type_sensitive(move):
            continue
        snapshot = turn.side(side)
        target = context.species(snapshot.target or turn.side(opponent(side)).active)
        if target is None:
            return math.nan
        attacks += 1
        resisted += chart.effectiveness(attack_type(move), target.types) < 1
    return resisted / attacks if attacks else 0.0


def _type_sensitive(move: Move) -> bool:
    return damage_kind(move) is DamageKind.STANDARD and move.name not in UTILITY_ATTACKS


def move_features(chart: TypeChart, turn: int | None = None) -> list[AtTurn]:
    def mv_res(context: BattleContext) -> float:
        return resisted_share(chart, context, "p2") - resisted_share(
            chart, context, "p1"
        )

    return [AtTurn("mv_res", mv_res, turn)]
