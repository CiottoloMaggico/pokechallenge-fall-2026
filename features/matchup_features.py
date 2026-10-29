from __future__ import annotations

import math
from collections.abc import Callable, Iterable

from domain.battle import BattleContext, LeadState
from domain.damage import STAGE_MULTIPLIER, Combatant, Field, expected_damage
from domain.typechart import TypeChart
from features.simulation_features import is_attack, stab_moves

NO_KO = 99
PARALYSIS_SPEED = 0.25
MAX_STAGE = 6


def active_duel(chart: TypeChart) -> Callable[[BattleContext], float]:
    def duel(context: BattleContext) -> float:
        p1_side, p2_side = context.side("p1"), context.side("p2")
        p1, p2 = p1_side.active, p2_side.active
        if p1.pokemon.fainted or p2.pokemon.fainted:
            return 0.0
        weather = context.battle.weather
        turns_p1 = _hits_to_ko(p1, p2, chart, weather, p2_side.screens)
        turns_p2 = _hits_to_ko(p2, p1, chart, weather, p1_side.screens)
        if turns_p1 != turns_p2:
            return 1.0 if turns_p1 < turns_p2 else -1.0
        speed_p1, speed_p2 = _speed(p1), _speed(p2)
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
    shown = [m for m in attacker.pokemon.known_moves.values() if is_attack(m)]
    best = 0.0
    for move in (*shown, *stab_moves(attacker.pokemon.info.types)):
        mean = expected_damage(
            a, d, move, chart, conditions=conditions, include_accuracy=True
        )
        if mean is not None:
            best = max(best, mean / d.max_hp)
    best = min(best, 1.0)
    return math.ceil(defender.pokemon.hp_pct / best) if best > 0 else NO_KO


def _speed(lead: LeadState) -> float:
    pokemon = lead.pokemon
    stage = max(-MAX_STAGE, min(MAX_STAGE, lead.boosts.get("spe", 0)))
    value = pokemon.stat("spe") * STAGE_MULTIPLIER[stage] / 100
    return value * (PARALYSIS_SPEED if pokemon.status == "par" else 1.0)
