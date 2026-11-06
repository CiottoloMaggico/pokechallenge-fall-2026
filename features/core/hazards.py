"""Spikes, Rapid Spin and phazing, counted up to the sampled turn.

In Gen 2 Spikes have a single layer, so a Spikes use *places* them only
when the opponent's side had none; a Rapid Spin *clears* them only when
the user's side had them. Both are read from the replayed state, turn by
turn. Counts become ``(P1 - P2) / (P1 + P2)``, in [-1, 1] (0 when
neither player did it).
"""

from __future__ import annotations

from domain.battle import BattleContext, SideState, TurnState
from features.core.turn import AtTurn, opponent, share_difference

SPIKES = "spikes"
RAPID_SPIN = "rapidspin"
PHAZING_MOVES = frozenset({"roar", "whirlwind"})


def _move_name(snapshot: SideState) -> str | None:
    return snapshot.move.name if snapshot.move is not None else None


def _transitions(
    context: BattleContext,
) -> list[tuple[TurnState | None, TurnState]]:
    previous: list[TurnState | None] = [None, *context.history[:-1]]
    return list(zip(previous, context.history, strict=True))


def _had_spikes(snapshot: TurnState | None, side: str) -> bool:
    return snapshot is not None and snapshot.side(side).spikes


def spikes_placed(context: BattleContext, side: str) -> float:
    """Times ``side`` laid Spikes on a clean opposing field."""
    other = opponent(side)
    return sum(
        _move_name(cur.side(side)) == SPIKES
        and not _had_spikes(prev, other)
        and cur.side(other).spikes
        for prev, cur in _transitions(context)
    )


def spikes_spun(context: BattleContext, side: str) -> float:
    """Times ``side`` removed Spikes from its own field with Rapid Spin."""
    return sum(
        _move_name(cur.side(side)) == RAPID_SPIN
        and _had_spikes(prev, side)
        and not cur.side(side).spikes
        for prev, cur in _transitions(context)
    )


def phazing_uses(context: BattleContext, side: str) -> float:
    """Roar / Whirlwind uses by ``side``."""
    return sum(_move_name(t.side(side)) in PHAZING_MOVES for t in context.history)


def spikes_on_field(context: BattleContext) -> float:
    """+1 only P2's side has Spikes, -1 only P1's, 0 both or neither."""
    last = context.last
    return float(last.p2.spikes) - float(last.p1.spikes)


def hazard_features(turn: int | None = None) -> list[AtTurn]:
    """The four Spikes / phazing features, sampled at ``turn``."""
    return [
        AtTurn("spikes_placed", share_difference(spikes_placed), turn),
        AtTurn("spikes_spun", share_difference(spikes_spun), turn),
        AtTurn("spikes_on_field", spikes_on_field, turn),
        AtTurn("phazing", share_difference(phazing_uses), turn),
    ]
