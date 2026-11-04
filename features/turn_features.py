"""Building blocks for features measured at a chosen turn.

Every concrete feature sees the battle *as if the timeline stopped after
turn T* (``BattleContext.upto``): the final state, the history and every
counter use only the first T turns. ``turn=None`` means the last
recorded turn (turn 30 for most battles). Column names do not depend on
the turn, so the same model can be compared across turns.

Sign convention: a two-player feature is positive when P1 is ahead
and negative when P1 is behind. With quantities good to have (HP,
Pokémon alive) it is ``value(P1) - value(P2)``; with quantities bad to
have (statuses, sleep) it is ``value(P2) - value(P1)``. Each value is
normalised to [0, 1] first.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from domain.battle import BattleContext
from feature_set import Feature

SideValue = Callable[[BattleContext, str], float]
ContextValue = Callable[[BattleContext], float]


def check_turn(turn: int | None) -> int | None:
    if turn is not None and turn < 1:
        raise ValueError(f"turn must be >= 1 or None, got {turn}")
    return turn


class AtTurn(Feature[BattleContext]):
    """A one-column feature computed on the battle truncated at ``turn``.

    The value depends only on the battle, so it is remembered in the
    replayed battle (``BattleContext.cached``) and reused by every fold
    and by every copy of the feature set.
    """

    def __init__(self, name: str, fn: ContextValue, turn: int | None = None) -> None:
        super().__init__(name)
        self._name = name
        self._fn = fn
        self.turn = check_turn(turn)

    @property
    def output_names(self) -> tuple[str, ...]:
        return (self._name,)

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        context = battle.upto(self.turn)
        key = (self._name, self._fn)  # the function survives set copies
        value = context.cached(key, lambda: float(self._fn(context)))
        return {self._name: value}


def side_difference(per_side: SideValue, bad: bool = False) -> ContextValue:
    """``per_side(P1) - per_side(P2)``; reversed when the quantity is ``bad``."""
    sign = -1.0 if bad else 1.0

    def difference(context: BattleContext) -> float:
        return sign * (per_side(context, "p1") - per_side(context, "p2"))

    return difference


def share_difference(per_side: SideValue, bad: bool = False) -> ContextValue:
    """``(a - b) / (a + b)`` of two counts, in [-1, 1]; 0 if both are 0.

    Reversed (``(b - a) / (a + b)``) when the counted thing is ``bad``.
    """
    sign = -1.0 if bad else 1.0

    def difference(context: BattleContext) -> float:
        a, b = per_side(context, "p1"), per_side(context, "p2")
        return sign * (a - b) / (a + b) if a + b else 0.0

    return difference


def opponent(side: str) -> str:
    return "p2" if side == "p1" else "p1"
