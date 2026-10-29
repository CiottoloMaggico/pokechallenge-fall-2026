"""Target encoding of the species P2 has revealed (uses the labels).

For each species, the share of training battles P1 won when P2's team
contained it, smoothed towards the overall win rate with ``m`` pseudo
battles. ``te_p2_mean`` is the mean over P2's revealed species minus the
overall rate: positive when P2 shows species P1 usually beats.

The gym fits it inside each fold and gives the model out-of-fold values
on the training rows (``FeatureSet.fit_transform``), so a battle never
sees its own label.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from domain.battle import BattleContext
from feature_set import Feature
from features.turn_features import check_turn

DEFAULT_SMOOTHING = 30.0


class OpponentSpeciesEncoding(Feature[BattleContext]):
    requires_fit = True
    uses_labels = True

    def __init__(
        self, turn: int | None = None, smoothing: float = DEFAULT_SMOOTHING
    ) -> None:
        super().__init__("opponent_species_encoding")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self.turn = check_turn(turn)
        self.smoothing = smoothing
        self.prior = 0.5
        self.encoding: dict[str, float] = {}

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("te_p2_mean",)

    def _fit(
        self, battles: Sequence[BattleContext], labels: Sequence[int] | None
    ) -> None:
        if labels is None:
            raise ValueError("the encoding needs the training labels")
        self.prior = sum(labels) / len(labels)
        totals: dict[str, list[int]] = {}
        for context, label in zip(battles, labels, strict=True):
            for name in context.side("p2").team:
                counts = totals.setdefault(name, [0, 0])
                counts[0] += label
                counts[1] += 1
        m = self.smoothing
        self.encoding = {
            name: (wins + m * self.prior) / (n + m)
            for name, (wins, n) in totals.items()
        }

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        team = battle.upto(self.turn).side("p2").team
        values = [self.encoding.get(name, self.prior) for name in team]
        mean = sum(values) / len(values) if values else self.prior
        return {"te_p2_mean": mean - self.prior}


class TeamEncoding(Feature[BattleContext]):
    """Win rate of P1's exact team and of the species pairs on the field.

    * ``te_p1_team``: share of training battles P1 won with exactly this
      team (teams repeat a lot in the data), smoothed with ``m_team``.
    * ``te_pair_mean``: mean over (P1 species, P2 revealed species) pairs
      of the share of training battles P1 won with that pair, smoothed
      with ``m_pair``.

    Both are minus the overall win rate, in [-0.5, 0.5]: positive when
    the line-up usually favours P1. Fitted inside each fold, out of fold
    on the training rows.
    """

    requires_fit = True
    uses_labels = True

    def __init__(
        self,
        turn: int | None = None,
        team_smoothing: float = 10.0,
        pair_smoothing: float = 20.0,
    ) -> None:
        super().__init__("team_encoding")
        if team_smoothing <= 0 or pair_smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self.turn = check_turn(turn)
        self.team_smoothing = team_smoothing
        self.pair_smoothing = pair_smoothing
        self.prior = 0.5
        self.team: dict[tuple[str, ...], float] = {}
        self.pair: dict[tuple[str, str], float] = {}

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("te_p1_team", "te_pair_mean")

    def _fit(
        self, battles: Sequence[BattleContext], labels: Sequence[int] | None
    ) -> None:
        if labels is None:
            raise ValueError("the encoding needs the training labels")
        self.prior = sum(labels) / len(labels)
        teams: dict[tuple[str, ...], list[int]] = {}
        pairs: dict[tuple[str, str], list[int]] = {}
        for context, label in zip(battles, labels, strict=True):
            team = _p1_team(context)
            _count(teams, team, label)
            for p1 in team:
                for p2 in sorted(context.side("p2").team):
                    _count(pairs, (p1, p2), label)
        self.team = self._smooth(teams, self.team_smoothing)
        self.pair = self._smooth(pairs, self.pair_smoothing)

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        context = battle.upto(self.turn)
        team = _p1_team(context)
        values = [
            self.pair.get((p1, p2), self.prior)
            for p1 in team
            for p2 in sorted(context.side("p2").team)
        ]
        pair_mean = sum(values) / len(values) if values else self.prior
        return {
            "te_p1_team": self.team.get(team, self.prior) - self.prior,
            "te_pair_mean": pair_mean - self.prior,
        }

    def _smooth(self, totals: Mapping[Any, list[int]], m: float) -> dict[Any, float]:
        return {
            key: (wins + m * self.prior) / (n + m) for key, (wins, n) in totals.items()
        }


def _p1_team(context: BattleContext) -> tuple[str, ...]:
    return tuple(sorted(context.side("p1").team))


def _count(totals: dict[Any, list[int]], key: Any, label: int) -> None:
    counts = totals.setdefault(key, [0, 0])
    counts[0] += label
    counts[1] += 1
