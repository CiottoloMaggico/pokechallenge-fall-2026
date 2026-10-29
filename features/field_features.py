"""Situation of the two active Pokémon at the sampled turn: boosts and effects.

Signs: positive when P1 is ahead (more boosts, a Substitute up, the
opponent trapped).
"""

from __future__ import annotations

from collections.abc import Mapping

from domain.battle import BattleContext
from features.turn_features import AtTurn

MAX_BOOST_SUM = 30  # five stats, six stages each
TRAPPING_EFFECTS = frozenset(
    {"trapped", "wrap", "bind", "clamp", "firespin", "whirlpool"}
)


def boost_sum(context: BattleContext) -> float:
    """Sum of P1's active stat stages minus P2's, over 30.

    The Speed drop that comes with Curse (+1 Atk, +1 Def, -1 Spe) is not
    counted: with Attack and Defense both raised a lower Speed is the
    price of the setup, not a loss.
    """
    last = context.last
    p1, p2 = (_boost_total(side.boosts) for side in (last.p1, last.p2))
    return (p1 - p2) / MAX_BOOST_SUM


def _boost_total(boosts: Mapping[str, int]) -> int:
    total = sum(boosts.values())
    cursed = boosts.get("atk", 0) > 0 and boosts.get("def", 0) > 0
    speed = boosts.get("spe", 0)
    return total - speed if cursed and speed < 0 else total


def substitute_up(context: BattleContext) -> float:
    """+1 only P1 has a Substitute, -1 only P2, 0 both or neither."""
    last = context.last
    return float("substitute" in last.p1.effects) - float(
        "substitute" in last.p2.effects
    )


def trapped(context: BattleContext) -> float:
    """+1 only P2's active is trapped, -1 only P1's, 0 both or neither."""
    last = context.last
    p1, p2 = (bool(TRAPPING_EFFECTS & set(s.effects)) for s in (last.p1, last.p2))
    return float(p2) - float(p1)


def field_features(turn: int | None = None) -> list[AtTurn]:
    return [
        AtTurn("bst_sum", boost_sum, turn),
        AtTurn("eff_substitute", substitute_up, turn),
        AtTurn("eff_trapped", trapped, turn),
    ]
