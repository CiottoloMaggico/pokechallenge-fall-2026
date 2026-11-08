"""Material left on the field at the sampled turn.

Pokémon of P2 not revealed yet are assumed alive, healthy and at full
HP; P1's whole team is known from the start. Counts are divided by the
team size (6), so per-player values lie in [0, 1] and differences in
[-1, 1]: positive when P1 is ahead (fewer statuses or sleepers count
as being ahead).
"""

from __future__ import annotations

from domain.battle import NO_STATUS, BattleContext, PlayerState, PokemonState
from domain.moves import Move
from features.core.turn import AtTurn, side_difference

SLEEP = "slp"
TEAM_SIZE = 6
# share of the maximum HP restored by each recovery move (Gen 2)
RECOVERY_MOVES = {
    "recover": 0.5,
    "softboiled": 0.5,
    "milkdrink": 0.5,
    "moonlight": 0.5,
    "morningsun": 0.5,
    "synthesis": 0.5,
    "rest": 1.0,
}


def _player(context: BattleContext, side: str) -> PlayerState:
    return context.side(side)


def _unrevealed(player: PlayerState) -> int:
    return player.team_size - len(player.team)


def _alive(player: PlayerState) -> list[PokemonState]:
    return [p for p in player.team.values() if not p.fainted]


def hp_share(context: BattleContext, side: str) -> float:
    """Current HP of the team over its maximum (fractions, unrevealed full)."""
    player = _player(context, side)
    return player.hp_total / player.team_size


def alive_share(context: BattleContext, side: str) -> float:
    """Pokémon not knocked out over the team size."""
    player = _player(context, side)
    return player.n_alive / player.team_size


def statused_share(context: BattleContext, side: str) -> float:
    """Alive Pokémon with a major status over the team size."""
    player = _player(context, side)
    return player.n_statused / player.team_size


def healthy_share(context: BattleContext, side: str) -> float:
    """Alive Pokémon without a status over the team size."""
    player = _player(context, side)
    healthy = sum(p.status == NO_STATUS for p in _alive(player))
    return (healthy + _unrevealed(player)) / player.team_size


def rest_sleep_share(context: BattleContext, side: str) -> float:
    """Alive Pokémon asleep because they used Rest, over the team size."""
    player = _player(context, side)
    asleep = sum(p.status == SLEEP and p.asleep_by_rest for p in _alive(player))
    return asleep / player.team_size


def other_sleep_share(context: BattleContext, side: str) -> float:
    """Alive Pokémon put to sleep by the opponent, over the team size."""
    player = _player(context, side)
    asleep = sum(p.status == SLEEP and not p.asleep_by_rest for p in _alive(player))
    return asleep / player.team_size


def p1_mean_hp_alive(context: BattleContext) -> float:
    """Mean HP fraction of P1's Pokémon still alive (0 if none)."""
    alive = _alive(_player(context, "p1"))
    return sum(p.hp_pct for p in alive) / len(alive) if alive else 0.0


def p2_revealed_share(context: BattleContext) -> float:
    """Share of P2's team revealed so far."""
    player = _player(context, "p2")
    return len(player.team) / player.team_size


def hp_healed(context: BattleContext) -> float:
    """HP regained by P1's active Pokémon minus P2's, over the turns so far.

    Sums the turn-to-turn increases of the active Pokémon's HP (Leftovers,
    drain, ...), skipping turns with a switch. A recovery move counts
    with what it restored even when a slower opponent then hit back in
    the same turn: half the maximum HP (Rest: all of it), capped by the
    HP that was missing. In fractions of a Pokémon, divided by 6.
    """
    healed = {"p1": 0.0, "p2": 0.0}
    for before, after in zip(context.history, context.history[1:], strict=False):
        for side in healed:
            now, was = after.side(side), before.side(side)
            if not now.switched:
                healed[side] += _healed(was.hp_pct, now.hp_pct, now.status, now.move)
    return (healed["p1"] - healed["p2"]) / TEAM_SIZE


def hp_dealt(context: BattleContext) -> float:
    """HP taken from the opponent's active, P1's total vs P2's, in [-1, 1].

    Sums the turn-to-turn HP drops of the opposing active Pokémon while it
    stayed in (attacks, recoil, residual damage), plus the HP it had when
    it fainted unseen (a hidden P2 knockout shows the replacement at once),
    then ``(P1 - P2) / (P1 + P2)``: positive when P1 dealt more damage.
    """
    dealt = {"p1": 0.0, "p2": 0.0}
    for before, after in zip(context.history, context.history[1:], strict=False):
        for side, other in (("p1", "p2"), ("p2", "p1")):
            now, was = after.side(other), before.side(other)
            if not now.switched:
                dealt[side] += max(0.0, was.hp_pct - now.hp_pct)
            elif now.n_fainted > was.n_fainted:
                dealt[side] += was.hp_pct
    total = dealt["p1"] + dealt["p2"]
    return (dealt["p1"] - dealt["p2"]) / total if total else 0.0


def _healed(before: float, after: float, status: str, move: Move | None) -> float:
    if after <= 0:
        return 0.0
    gained = max(0.0, after - before)
    if move is None or move.name not in RECOVERY_MOVES:
        return gained
    if move.name == "rest" and status != SLEEP:
        return gained
    if move.name != "rest" and status in ("slp", "frz"):
        return gained
    restored = min(RECOVERY_MOVES[move.name], 1.0 - before)
    return max(gained, restored)


def material_features(turn: int | None = None) -> list[AtTurn]:
    """The eight material features, sampled at ``turn``."""
    return [
        AtTurn("material_hp", side_difference(hp_share), turn),
        AtTurn("material_alive", side_difference(alive_share), turn),
        AtTurn("material_statused", side_difference(statused_share, bad=True), turn),
        AtTurn("material_healthy", side_difference(healthy_share), turn),
        AtTurn("p1_mean_hp_alive", p1_mean_hp_alive, turn),
        AtTurn("material_asleep_rest", side_difference(rest_sleep_share, bad=True), turn),
        AtTurn("material_asleep_other", side_difference(other_sleep_share, bad=True), turn),
        AtTurn("p2_revealed", p2_revealed_share, turn),
    ]
