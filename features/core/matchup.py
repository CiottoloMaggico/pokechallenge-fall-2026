"""Matchups between the Pokémon of the two players at the sampled turn.

* ``act_duel``: who wins the duel between the two actives (best move
  among the revealed attacks and a power-80 STAB move, current stages,
  weather and screens; fewer hits to KO wins, speed breaks ties): +1 P1,
  -1 P2, 0 if equal or if one has fainted.
* ``act_ttk_adv``: the same duel as a margin: turns P2's active needs to
  knock out P1's minus the turns P1's needs (asleep / frozen turns
  included, half a turn to the faster one), over 20, in [-1, 1].
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping

from domain.battle import BattleContext, LeadState, PlayerState, PokemonState
from domain.damage import STAGE_MULTIPLIER, Combatant, DamageCalculator, Field
from domain.typechart import TypeChart
from features.core.simulation import is_attack, stab_moves, turns_lost

NO_KO = 99
PARALYSIS_SPEED = 0.25
MAX_STAGE = 6
TTK_CAP = 20.0
FASTER_BONUS = 0.5


def active_duel(chart: TypeChart) -> Callable[[BattleContext], float]:
    def duel(context: BattleContext) -> float:
        p1_side, p2_side = context.side("p1"), context.side("p2")
        p1, p2 = p1_side.active, p2_side.active
        if p1.pokemon.fainted or p2.pokemon.fainted:
            return 0.0
        if not (p1.pokemon.info.has_stats and p2.pokemon.info.has_stats):
            return math.nan
        weather = context.battle.weather
        turns_p1 = _hits_to_ko(p1, p2, chart, weather, p2_side.screens)
        turns_p2 = _hits_to_ko(p2, p1, chart, weather, p1_side.screens)
        if turns_p1 != turns_p2:
            return 1.0 if turns_p1 < turns_p2 else -1.0
        speed_p1, speed_p2 = active_speed(p1), active_speed(p2)
        return float((speed_p1 > speed_p2) - (speed_p1 < speed_p2))

    return duel


def _hits_to_ko(
    attacker: LeadState,
    defender: LeadState,
    chart: TypeChart,
    weather: str | None = None,
    defender_screens: Iterable[str] = (),
) -> int:
    a = Combatant.from_state(attacker.pokemon, attacker.boosts)
    d = Combatant.from_state(defender.pokemon, defender.boosts)
    conditions = Field.of(weather, defender_screens)
    damage = DamageCalculator(chart)
    shown = [m for m in attacker.pokemon.known_moves.values() if is_attack(m)]
    best = 0.0
    for move in (*shown, *stab_moves(attacker.pokemon.info.types)):
        mean = damage.expected(a, d, move, conditions, with_accuracy=True)
        if mean is not None:
            best = max(best, mean / d.max_hp)
    best = min(best, 1.0)
    return math.ceil(defender.pokemon.hp_pct / best) if best > 0 else NO_KO


def active_speed(lead: LeadState) -> float:
    pokemon = lead.pokemon
    stage = max(-MAX_STAGE, min(MAX_STAGE, lead.boosts.get("spe", 0)))
    value = pokemon.stat("spe") * STAGE_MULTIPLIER[stage] / 100
    return value * (PARALYSIS_SPEED if pokemon.status == "par" else 1.0)


def ttk_advantage(chart: TypeChart) -> Callable[[BattleContext], float]:
    """``act_ttk_adv`` (see the module docstring)."""

    def advantage(context: BattleContext) -> float:
        p1_side, p2_side = context.side("p1"), context.side("p2")
        p1, p2 = p1_side.active, p2_side.active
        if p1.pokemon.fainted or p2.pokemon.fainted:
            return 0.0
        if not (p1.pokemon.info.has_stats and p2.pokemon.info.has_stats):
            return math.nan
        weather = context.battle.weather
        share_12 = best_share(
            p1.pokemon,
            p2.pokemon,
            chart,
            p1.boosts,
            p2.boosts,
            weather,
            p2_side,
        )
        share_21 = best_share(
            p2.pokemon,
            p1.pokemon,
            chart,
            p2.boosts,
            p1.boosts,
            weather,
            p1_side,
        )
        turns_12 = turns_needed(p2.pokemon.hp_pct, share_12) + turns_lost(p1.pokemon)
        turns_21 = turns_needed(p1.pokemon.hp_pct, share_21) + turns_lost(p2.pokemon)
        speed = active_speed(p1) - active_speed(p2)
        faster = FASTER_BONUS * ((speed > 0) - (speed < 0))
        margin = turns_21 - turns_12 + faster
        return max(-1.0, min(1.0, margin / TTK_CAP))

    return advantage


def best_share(
    attacker: PokemonState,
    defender: PokemonState,
    chart: TypeChart,
    attacker_boosts: Mapping[str, int] | None = None,
    defender_boosts: Mapping[str, int] | None = None,
    weather: str | None = None,
    defender_side: PlayerState | None = None,
) -> float:
    """Best expected damage per turn, as a share of the defender's max HP.

    The attacks are the attacker's revealed attacking moves; a power-80 STAB move per type is
    always a candidate.
    """
    a = Combatant.from_state(attacker, _stages(attacker_boosts))
    d = Combatant.from_state(defender, _stages(defender_boosts))
    screens = defender_side.screens if defender_side is not None else ()
    conditions = Field.of(weather, screens)
    damage = DamageCalculator(chart)
    shown = [m for m in attacker.known_moves.values() if is_attack(m)]
    best = 0.0
    for move in (*shown, *stab_moves(attacker.info.types)):
        mean = damage.expected(a, d, move, conditions, with_accuracy=True)
        if mean is not None:
            best = max(best, mean / d.max_hp)
    return min(best, 1.0)


def turns_needed(hp: float, share: float) -> float:
    return min(TTK_CAP, hp / share) if share > 0 else TTK_CAP


def _stages(boosts: Mapping[str, int] | None) -> dict[str, int]:
    return {k: max(-MAX_STAGE, min(MAX_STAGE, v)) for k, v in (boosts or {}).items()}
