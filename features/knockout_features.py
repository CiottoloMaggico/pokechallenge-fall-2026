"""Features about knockouts the records may hide (see ``domain.transitions``).

* ``p2_hidden_ko_soft``: expected number of hidden P2 knockouts, over 6.
  For every P2 switch whose Pokémon never comes back, a probability of
  knockout is computed with Bayes' rule from two logistic models learned
  on P1's turns, where faints are recorded (no labels involved)::

      P(KO | switch, x) = P(KO | x) / (P(KO | x) + (1 - P(KO | x)) P(switch | x))

  with ``x`` = HP before the turn and expected damage of the opposing
  move (mean, max, their margins over the HP, whether it attacked). The
  opposing move counts only if that Pokémon was already on the field
  (the records repeat an old move on a switch-in); switches forced by
  Roar / Whirlwind are skipped. In [0, 1].
* ``p2_hidden_ko_det``: expected number of hidden P2 knockouts according
  to the learned detector used to repair the battle
  (the context's ``repair``): the sum of its probabilities over P2's
  switches, over 6. In [0, 1].
* ``p2_hidden_ko_unsure``: the same sum restricted to the switches the
  repair did not mark as knockouts: the knockouts the material features
  cannot see yet. In [0, 1].
* ``ghost_ko_50``: revealed Pokémon off the field at or below half HP,
  not fainted (after the hidden-knockout repair), P2's count minus P1's,
  over 6: positive when P2 has more such probable losses. In [-1, 1].
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from domain.battle import BattleContext
from domain.damage import Combatant, DamageCalculator
from domain.moves import Move
from domain.typechart import TypeChart
from feature_set import Feature
from features.turn_features import check_turn

TEAM_SIZE = 6
GHOST_HP = 0.5
DEFAULT_LEVEL = 100
MAX_STAGE = 6
FAINTED = "fnt"
PHAZING_MOVES = frozenset({"roar", "whirlwind"})

Record = Mapping[str, Any]


def ghost_knockouts(context: BattleContext) -> float:
    """``ghost_ko_50`` (see the module docstring)."""
    counts = {}
    for side in ("p1", "p2"):
        final = context.last.side(side).active
        counts[side] = sum(
            1
            for pokemon in context.side(side).team.values()
            if pokemon.seen
            and pokemon.name != final
            and not pokemon.fainted
            and 0 < pokemon.hp_pct <= GHOST_HP
        )
    return (counts["p2"] - counts["p1"]) / TEAM_SIZE


def expected_hidden_knockouts(context: BattleContext) -> float:
    """``p2_hidden_ko_det`` (see the module docstring)."""
    return min(1.0, sum(_detector_probabilities(context)) / TEAM_SIZE)


def unsure_hidden_knockouts(context: BattleContext) -> float:
    """``p2_hidden_ko_unsure`` (see the module docstring)."""
    probabilities = context.repair.probabilities(context.record)
    repaired = context.repair.repairs(context.record)
    unsure = [p for index, p in probabilities.items() if index not in repaired]
    return min(1.0, sum(unsure) / TEAM_SIZE)


def _detector_probabilities(context: BattleContext) -> list[float]:
    """Probabilities of the context's repair; none with ``NoRepair``."""
    return list(context.repair.probabilities(context.record).values())


class HiddenKnockoutProbability(Feature[BattleContext]):
    """``p2_hidden_ko_soft`` at the sampled turn (see the module docstring)."""

    requires_fit = True

    def __init__(self, chart: TypeChart, turn: int | None = None) -> None:
        super().__init__("hidden_knockout_probability")
        self.chart = chart
        self.turn = check_turn(turn)
        self.knockout_model: LogisticRegression | None = None
        self.switch_model: LogisticRegression | None = None

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("p2_hidden_ko_soft",)

    def _fit(
        self, battles: Sequence[BattleContext], labels: Sequence[int] | None
    ) -> None:
        rows, knocked_out, switched = [], [], []
        for context in battles:
            for x, ko, switch in self._p1_turns(context):
                rows.append(x)
                knocked_out.append(ko)
                switched.append(switch)
        x_all, ko_all, sw_all = (
            np.array(rows),
            np.array(knocked_out),
            np.array(switched),
        )
        self.knockout_model = LogisticRegression(max_iter=3000).fit(x_all, ko_all)
        alive = ko_all == 0
        self.switch_model = LogisticRegression(max_iter=3000).fit(
            x_all[alive], sw_all[alive]
        )

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        context = battle.at_turn(self.turn)
        probabilities = self.knockout_probabilities(context)
        return {"p2_hidden_ko_soft": sum(probabilities.values()) / TEAM_SIZE}

    def knockout_probabilities(self, context: BattleContext) -> dict[str, float]:
        """P2 Pokémon -> probability that it was knocked out unseen."""
        if self.knockout_model is None or self.switch_model is None:
            raise RuntimeError("fit the feature first")
        timeline = context.record["battle_timeline"]
        levels = _p1_levels(context.record)
        result: dict[str, float] = {}
        for i in range(1, len(timeline)):
            before = timeline[i - 1]["p2_pokemon_state"]
            after = timeline[i]["p2_pokemon_state"]
            if before["name"] == after["name"] or before["status"] == FAINTED:
                continue
            if any(
                t["p2_pokemon_state"]["name"] == before["name"] for t in timeline[i:]
            ):
                continue
            move = _move_used(timeline, i, "p1")
            if move is not None and move["name"] in PHAZING_MOVES:
                continue
            attacker = timeline[i]["p1_pokemon_state"]
            fighters = (
                self._combatant(context, attacker, levels.get(attacker["name"])),
                self._combatant(context, before, DEFAULT_LEVEL),
            )
            if _is_attack(move) and None in fighters:
                return {before["name"]: math.nan}
            mean, worst = self._damage(*fighters, move)
            x = np.array([_row(before["hp_pct"], mean, worst, _is_attack(move))])
            p_ko = float(self.knockout_model.predict_proba(x)[0, 1])
            p_switch = float(self.switch_model.predict_proba(x)[0, 1])
            evidence = p_ko + (1 - p_ko) * p_switch
            p = p_ko / evidence if evidence > 0 else 0.0
            result[before["name"]] = max(result.get(before["name"], 0.0), p)
        return result

    def _p1_turns(self, context: BattleContext) -> list[tuple[list[float], int, int]]:
        timeline = context.record["battle_timeline"]
        levels = _p1_levels(context.record)
        rows = []
        for i in range(1, len(timeline)):
            before = timeline[i - 1]["p1_pokemon_state"]
            after = timeline[i]["p1_pokemon_state"]
            if before["status"] == FAINTED or before["hp_pct"] <= 0:
                continue
            move = _move_used(timeline, i, "p2")
            if move is not None and move["name"] in PHAZING_MOVES:
                continue
            attacker = timeline[i]["p2_pokemon_state"]
            fighters = (
                self._combatant(context, attacker, DEFAULT_LEVEL),
                self._combatant(context, before, levels.get(before["name"])),
            )
            if _is_attack(move) and None in fighters:
                continue  # its damage is unknown, so it is no training example
            mean, worst = self._damage(*fighters, move)
            knocked_out = int(
                after["name"] == before["name"] and after["status"] == FAINTED
            )
            switched = int(after["name"] != before["name"]) if not knocked_out else 0
            x = _row(before["hp_pct"], mean, worst, _is_attack(move))
            rows.append((x, knocked_out, switched))
        return rows

    @staticmethod
    def _combatant(
        context: BattleContext, state: Record, level: int | None
    ) -> Combatant | None:
        species = context.species(state["name"])
        if species is None:
            return None
        boosts = {
            k: max(-MAX_STAGE, min(MAX_STAGE, v))
            for k, v in (state.get("boosts") or {}).items()
        }
        status = state["status"] if state["status"] != FAINTED else "nostatus"
        return Combatant(
            species, level=level or DEFAULT_LEVEL, boosts=boosts, status=status
        )

    def _damage(
        self,
        attacker: Combatant | None,
        defender: Combatant | None,
        move: Record | None,
    ) -> tuple[float, float]:
        if attacker is None or defender is None or not _is_attack(move):
            return 0.0, 0.0
        assert move is not None
        damage = DamageCalculator(self.chart)
        result = damage.roll(attacker, defender, Move.from_details(move))
        if result is None:
            return 0.0, 0.0
        share = result.fraction_of(defender.max_hp)
        return share.mean, share.max


def _row(hp: float, mean: float, worst: float, attack: bool) -> list[float]:
    return [hp, mean, worst, mean - hp, worst - hp, float(attack), float(worst >= hp)]


def _move_used(timeline: Sequence[Record], i: int, side: str) -> Record | None:
    """``side``'s move at turn ``i``, None if its Pokémon just switched in.

    On a switch-in the records repeat the incoming Pokémon's old move.
    """
    name = timeline[i][f"{side}_pokemon_state"]["name"]
    if timeline[i - 1][f"{side}_pokemon_state"]["name"] != name:
        return None
    return timeline[i][f"{side}_move_details"]


def _is_attack(move: Record | None) -> bool:
    return move is not None and move["category"] != "status"


def _p1_levels(record: Record) -> dict[str, int]:
    return {d["name"]: d.get("level", DEFAULT_LEVEL) for d in record["p1_team_details"]}
