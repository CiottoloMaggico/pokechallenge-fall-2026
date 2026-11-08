"""Probability that each player switches out at the next turn.

Learned from the training battles (no labels): for every pair
(my active Pokémon, opposing active Pokémon) the share of turns after
which I switched out voluntarily, i.e. changed Pokémon without using a
move. Rare pairs are smoothed towards the rate of my active species,
which is smoothed towards the global rate::

    p(pair) = (switches + m * p(species)) / (turns + m)

A fainted active Pokémon must be replaced, so its probability is 1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from domain.battle import BattleContext, TurnState
from features.core.turn import AtTurn, check_turn, opponent, share_difference
from interfaces import Feature

DEFAULT_SMOOTHING = 5.0
MAX_TURNS = 30
SIDES = ("p1", "p2")


class NextSwitchProbability(Feature[BattleContext]):
    """``p1_p_switch`` and ``p2_p_switch`` at the sampled turn."""

    requires_fit = True

    def __init__(self, turn: int | None = None, smoothing: float = DEFAULT_SMOOTHING) -> None:
        super().__init__("next_switch_probability")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self.turn = check_turn(turn)
        self.smoothing = smoothing
        self.global_rate = 0.0
        self.species_rate: dict[str, float] = {}
        self.pair_rate: dict[tuple[str, str], float] = {}

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(f"{side}_p_switch" for side in SIDES)

    def _fit(self, battles: Sequence[BattleContext], labels: Sequence[int] | None) -> None:
        pairs: dict[tuple[str, str], list[int]] = {}
        for context in battles:
            for prev, cur in zip(context.history, context.history[1:], strict=False):
                for side in SIDES:
                    key = _decision(prev, side)
                    if key is not None:
                        counts = pairs.setdefault(key, [0, 0])
                        counts[0] += cur.side(side).voluntary_switch
                        counts[1] += 1
        species: dict[str, list[int]] = {}
        for (mine, _), (switches, turns) in pairs.items():
            counts = species.setdefault(mine, [0, 0])
            counts[0] += switches
            counts[1] += turns
        total_switches = sum(c[0] for c in species.values())
        total_turns = sum(c[1] for c in species.values())
        self.global_rate = total_switches / total_turns if total_turns else 0.0
        self.species_rate = {
            name: self._smooth(s, n, self.global_rate) for name, (s, n) in species.items()
        }
        self.pair_rate = {
            key: self._smooth(s, n, self.species_rate[key[0]]) for key, (s, n) in pairs.items()
        }

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        last = battle.at_turn(self.turn).last
        return {f"{side}_p_switch": self.probability(last, side) for side in SIDES}

    def probability(self, snapshot: TurnState, side: str) -> float:
        """Chance that ``side`` switches out after ``snapshot``."""
        key = _decision(snapshot, side)
        if key is None:
            return 1.0  # the active Pokémon fainted: a switch is forced
        if key in self.pair_rate:
            return self.pair_rate[key]
        return self.species_rate.get(key[0], self.global_rate)

    def _smooth(self, switches: int, turns: int, prior: float) -> float:
        return (switches + self.smoothing * prior) / (turns + self.smoothing)


def _decision(snapshot: TurnState, side: str) -> tuple[str, str] | None:
    """(my active, opposing active) when ``side`` can choose; None if fainted."""
    mine = snapshot.side(side)
    if mine.hp_pct <= 0:
        return None
    return mine.active, snapshot.side(opponent(side)).active


class SwitchPressure(NextSwitchProbability):
    """Who is being pushed to switch, from the same learned rates.

    * ``p_switch_diff``: P2's next-switch probability minus P1's at the
      sampled turn (positive when P2 is the one likely to switch out).
    * ``switch_excess``: voluntary switches beyond the expected ones,
      summed over the turns so far, P2's excess minus P1's, over 30.
    * ``switch_excess_all``: the same counting every switch, also those
      forced by a faint or by Roar / Whirlwind (so it also reflects the
      Pokémon each side has lost).

    ``outputs`` picks which of them to compute.
    """

    OUTPUTS = ("p_switch_diff", "switch_excess", "switch_excess_all")

    def __init__(
        self,
        turn: int | None = None,
        smoothing: float = DEFAULT_SMOOTHING,
        outputs: Sequence[str] = ("p_switch_diff", "switch_excess"),
    ) -> None:
        super().__init__(turn, smoothing)
        unknown = set(outputs) - set(self.OUTPUTS)
        if not outputs or unknown:
            raise ValueError(f"outputs must be a non-empty subset of {self.OUTPUTS}")
        self._feature_id = "switch_pressure"
        self.outputs = tuple(outputs)

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.outputs

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        context = battle.at_turn(self.turn)
        last = context.last
        voluntary = {side: 0.0 for side in SIDES}
        every = {side: 0.0 for side in SIDES}
        for prev, cur in zip(context.history, context.history[1:], strict=False):
            for side in SIDES:
                if _decision(prev, side) is not None:
                    expected = self.probability(prev, side)
                    voluntary[side] += cur.side(side).voluntary_switch - expected
                    every[side] += cur.side(side).switched - expected
        values = {
            "p_switch_diff": self.probability(last, "p2") - self.probability(last, "p1"),
            "switch_excess": (voluntary["p2"] - voluntary["p1"]) / MAX_TURNS,
            "switch_excess_all": (every["p2"] - every["p1"]) / MAX_TURNS,
        }
        return {name: values[name] for name in self.outputs}


def switches_count(context: BattleContext, side: str) -> float:
    """Total number of switches made by ``side`` up to the sampled turn."""
    return float(sum(t.side(side).switched for t in context.history))


def switch_count_feature(turn: int | None = None) -> AtTurn:
    """Share difference in [-1, 1] of switches: (P2 - P1) / (P1 + P2)."""
    return AtTurn("switch_share", share_difference(switches_count, bad=True), turn)
