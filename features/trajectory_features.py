"""How the battle got to the sampled turn: averages and trends over the turns.

Two players with the same material at turn 30 are not in the same spot if
one of them was ahead all game and the other just caught up. Per turn,
from the replayed history (with the hidden knockouts already applied):

* HP balance: P1's team HP minus P2's (unrevealed Pokémon at full HP),
  in Pokémon, over 6;
* knockout balance: P2's fainted Pokémon minus P1's, over 6.

Features (positive when P1 is ahead):

* ``traj_hp_mean``: mean HP balance over the turns so far, in [-1, 1];
* ``traj_ko_mean``: mean knockout balance over the turns so far, in [-1, 1];
* ``traj_hp_delta10``: HP balance now minus ten turns before (the first
  turn for shorter battles), over 2, in [-1, 1].
"""

from __future__ import annotations

from domain.battle import BattleContext

TEAM_SIZE = 6
WINDOW = 10


def _hp_balance(context: BattleContext) -> list[float]:
    return [(t.p1.hp_total - t.p2.hp_total) / TEAM_SIZE for t in context.history]


def _ko_balance(context: BattleContext) -> list[float]:
    return [(t.p2.n_fainted - t.p1.n_fainted) / TEAM_SIZE for t in context.history]


def hp_balance_mean(context: BattleContext) -> float:
    """``traj_hp_mean`` (see the module docstring)."""
    values = _hp_balance(context)
    return sum(values) / len(values)


def ko_balance_mean(context: BattleContext) -> float:
    """``traj_ko_mean`` (see the module docstring)."""
    values = _ko_balance(context)
    return sum(values) / len(values)


def hp_balance_delta(context: BattleContext) -> float:
    """``traj_hp_delta10`` (see the module docstring)."""
    values = _hp_balance(context)
    before = values[max(0, len(values) - 1 - WINDOW)]
    return (values[-1] - before) / 2
