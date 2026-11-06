"""Monte Carlo version of the rest-of-battle simulation (prototype).

:class:`features.core.simulation.BattleSimulation` plays each duel with
expected damage: a Pokémon needs ``ceil(hp / mean)`` hits. Here every duel
is played ``runs`` times with random outcomes instead:

* each attack misses with the move's inaccuracy, and is a critical hit with
  the move's critical rate (standard damage moves only);
* the damage is drawn uniformly over the 39 random rolls of the Gen 2
  formula (``DamageCalculator.roll`` gives the minimum and maximum);
* a speed tie is broken by a coin toss, not by the order of the arguments.

The attacker of each duel always uses the move with the best expected
damage, as in the deterministic simulation. The fitted parts (typical
moves of each species, naive Bayes guess of P2's hidden Pokémon) are
inherited from :class:`BattleSimulation`, so training folds only.

Outputs (positive when P1 comes out ahead, as their deterministic twins):

* ``mc_seen_left``: mean over runs of (P1 left - P2 left) / 6 against
  P2's revealed Pokémon;
* ``mc_full_win``: mean over runs of +1 (P1 wins), -1 (P2 wins), 0.

Each battle has its own seed, derived from its content fingerprint, so
the result does not depend on the order of the battles or on the process.
Values are not cached on the battle: they depend on the fitted state.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping

from domain.battle import BattleContext
from domain.damage import DamageCalculator
from domain.moves import DamageKind, Move, crit_rate, damage_kind
from domain.records import fingerprint
from domain.typechart import TypeChart
from features.core.simulation import (
    TEAM_SIZE,
    BattleSimulation,
    DuelSimulator,
    Fighter,
    outcome,
    stats_missing,
)

MC_OUTPUTS = ("mc_seen_left", "mc_full_win")
RUNS = 32
MAX_TURNS = 60
MIN_HP_LEFT = 0.01
SEED_DIGITS = 15


class MonteCarloDuels(DuelSimulator):
    """:class:`DuelSimulator` whose duels are sampled with ``rng``."""

    def __init__(self, chart: TypeChart, rng: random.Random) -> None:
        super().__init__(chart)
        self.rng = rng

    def _duel(
        self, a: Fighter, b: Fighter, cache: dict[tuple[str, str], tuple[float, float]]
    ) -> tuple[bool, float]:
        move_a = self._choice(a, b)
        move_b = self._choice(b, a)
        hp = {"a": a.hp, "b": b.hp}
        delay = {"a": a.delay, "b": b.delay}
        if a.combatant.stat("spe") == b.combatant.stat("spe"):
            first = ("a", "b") if self.rng.random() < 0.5 else ("b", "a")
        else:
            faster_a = a.combatant.stat("spe") > b.combatant.stat("spe")
            first = ("a", "b") if faster_a else ("b", "a")
        fighters = {"a": a, "b": b}
        moves = {"a": move_a, "b": move_b}
        for _ in range(MAX_TURNS):
            for side in first:
                other = "b" if side == "a" else "a"
                if delay[side] > 0:
                    delay[side] -= 1
                    continue
                damage = self._roll(fighters[side], fighters[other], moves[side])
                hp[other] -= damage
                if hp[other] <= 0:
                    return side == "a", max(MIN_HP_LEFT, hp[side])
        winner = "a" if hp["a"] >= hp["b"] else "b"
        return winner == "a", max(MIN_HP_LEFT, hp[winner])

    def _choice(self, attacker: Fighter, defender: Fighter) -> Move | None:
        """The move with the best expected damage, as in ``_best``."""
        best: Move | None = None
        best_mean = 0.0
        damage = DamageCalculator(self.chart)
        for move in attacker.moves:
            mean = damage.expected(
                attacker.combatant, defender.combatant, move, with_accuracy=True
            )
            if mean is not None and mean > best_mean:
                best, best_mean = move, mean
        return best

    def _roll(self, attacker: Fighter, defender: Fighter, move: Move | None) -> float:
        """Damage of one attack, as a share of the defender's max HP."""
        if move is None or self.rng.random() >= move.accuracy:
            return 0.0
        critical = damage_kind(move) is DamageKind.STANDARD and (
            self.rng.random() < crit_rate(move)
        )
        rolls = DamageCalculator(self.chart).roll(
            attacker.combatant,
            defender.combatant,
            move,
            crit=critical,
        )
        if rolls is None:
            return 0.0
        value = self.rng.uniform(rolls.min, rolls.max)
        return value / defender.combatant.max_hp


class MonteCarloSimulation(BattleSimulation):
    """``mc_seen_left`` and ``mc_full_win`` at the sampled turn."""

    def __init__(
        self, chart: TypeChart, turn: int | None = None, runs: int = RUNS
    ) -> None:
        super().__init__(chart, turn)
        if runs < 1:
            raise ValueError(f"runs must be >= 1, got {runs}")
        self._feature_id = "montecarlo_simulation"
        self.runs = runs

    @property
    def output_names(self) -> tuple[str, ...]:
        return MC_OUTPUTS

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        context = battle.at_turn(self.turn)
        p1, p2 = context.side("p1"), context.side("p2")
        team_a = [
            self._fighter(p, p1.active.boosts if p is p1.active.pokemon else None)
            for p in p1.team.values()
            if not p.fainted
        ]
        seen_b = [
            self._fighter(p, p2.active.boosts if p is p2.active.pokemon else None)
            for p in p2.team.values()
            if not p.fainted
        ]
        hidden = self.guess(list(p2.team), p2.team_size)
        full_b = seen_b + [
            self._fighter(context.new_pokemon(name)) for name in hidden
        ]
        if stats_missing(team_a, full_b):
            return {name: math.nan for name in MC_OUTPUTS}
        actives = (p1.active.pokemon.name, p2.active.pokemon.name)
        seed = int(fingerprint(battle.record)[:SEED_DIGITS], 16)
        seen_values: list[float] = []
        full_values: list[float] = []
        for run in range(self.runs):
            simulator = MonteCarloDuels(self.chart, random.Random(seed + run))
            left_a, left_b = simulator.simulate(team_a, seen_b, actives)
            seen_values.append((len(left_a) - len(left_b)) / TEAM_SIZE)
            win_a, win_b = simulator.simulate(team_a, full_b, actives)
            full_values.append(outcome(win_a, win_b))
        return {
            "mc_seen_left": sum(seen_values) / self.runs,
            "mc_full_win": sum(full_values) / self.runs,
        }
