"""Target encodings of species and teams (they use the labels).

``TeamEffect`` (at the end) credits each team with how much better than
expected its battles went.

Target encoding of the species P2 has revealed:

For each species, the share of training battles P1 won when P2's team
contained it, smoothed towards the overall win rate with ``m`` pseudo
battles. ``te_p2_mean`` is the mean over P2's revealed species minus the
overall rate: positive when P2 shows species P1 usually beats.

The gym fits it inside each fold and gives the model out-of-fold values
on the training rows (``FeatureSet.fit_transform``), so a battle never
sees its own label.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from domain.battle import BattleContext
from feature_set import Feature
from features.knockout_features import expected_hidden_knockouts
from features.material_features import alive_share, healthy_share, hp_share
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
        team = battle.at_turn(self.turn).side("p2").team
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
        context = battle.at_turn(self.turn)
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


class TeamEffect(Feature[BattleContext]):
    """How much better than expected the players behind both teams do.

    Teams are reused across battles (70% of the test battles use a P1
    team seen in training), so a team identifies (a group of) players.
    For each training battle a small logistic regression on the battle
    state (HP, Pokémon alive, healthy Pokémon, expected hidden
    knockouts) gives the expected outcome, and the residual (outcome
    minus expectation) is credited to P1's team and, with the opposite
    sign, to P2's team when it is known: all six revealed, or five
    revealed that belong to a single training team.

    * ``tm_team``: shrunk mean residual of P1's team minus that of P2's
      team (0 if unknown), over 2, in [-1, 1];
    * ``tm_pair``: shrunk mean residual of the battles between these two
      teams (repeated series between the same players), in [-1, 1].

    ``smoothing`` pseudo battles with residual 0 shrink rare teams. The
    gym fits it inside each fold, out of fold on the training rows.
    """

    requires_fit = True
    uses_labels = True
    OUTPUTS = ("tm_team", "tm_pair")

    def __init__(self, turn: int | None = None, smoothing: float = 1.0) -> None:
        super().__init__("team_effect")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self.turn = check_turn(turn)
        self.smoothing = smoothing
        self.links: dict[frozenset[str], set[frozenset[str]]] = {}
        self.sums: dict[Any, float] = {}
        self.counts: dict[Any, int] = {}
        self.base: Any = None

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.OUTPUTS

    def _fit(
        self, battles: Sequence[BattleContext], labels: Sequence[int] | None
    ) -> None:
        if labels is None:
            raise ValueError("the team effect needs the training labels")
        contexts = [battle.at_turn(self.turn) for battle in battles]
        y = np.asarray(labels)
        x = np.array([_state(context) for context in contexts])
        self.base = LogisticRegression(max_iter=2000).fit(x, y)
        residuals = y - self.base.predict_proba(x)[:, 1]
        self.links = _links(contexts)
        sums: dict[Any, float] = defaultdict(float)
        counts: dict[Any, int] = defaultdict(int)
        for context, r in zip(contexts, residuals, strict=True):
            mine, theirs = self._teams(context)
            sums[mine] += r
            counts[mine] += 1
            if theirs is not None:
                sums[theirs] -= r
                counts[theirs] += 1
                sums[mine, theirs] += r
                counts[mine, theirs] += 1
                sums[theirs, mine] -= r
                counts[theirs, mine] += 1
        self.sums, self.counts = dict(sums), dict(counts)

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        context = battle.at_turn(self.turn)
        mine, theirs = self._teams(context)
        own = self._effect(mine)
        other = self._effect(theirs) if theirs is not None else 0.0
        pair = self._effect((mine, theirs)) if theirs is not None else 0.0
        return {"tm_team": (own - other) / 2, "tm_pair": pair}

    def _effect(self, key: Any) -> float:
        return self.sums.get(key, 0.0) / (self.counts.get(key, 0) + self.smoothing)

    def _teams(
        self, context: BattleContext
    ) -> tuple[frozenset[str], frozenset[str] | None]:
        """P1's team and P2's team when it can be identified."""
        mine = frozenset(context.side("p1").team)
        revealed = frozenset(context.side("p2").team)
        if len(revealed) == context.side("p2").team_size:
            return mine, revealed
        if len(revealed) == context.side("p2").team_size - 1:
            candidates = self.links.get(revealed, set()) - {mine}
            if len(candidates) == 1:
                return mine, next(iter(candidates))
        return mine, None


def _links(
    contexts: Sequence[BattleContext],
) -> dict[frozenset[str], set[frozenset[str]]]:
    """Five-species subsets of every complete team -> the teams containing it."""
    links: dict[frozenset[str], set[frozenset[str]]] = defaultdict(set)
    for context in contexts:
        for side in ("p1", "p2"):
            player = context.side(side)
            team = frozenset(player.team)
            if len(team) == player.team_size:
                for subset in combinations(sorted(team), len(team) - 1):
                    links[frozenset(subset)].add(team)
    return dict(links)


def _state(context: BattleContext) -> list[float]:
    """Battle state for the expected outcome of the team effect."""
    return [
        hp_share(context, "p1") - hp_share(context, "p2"),
        alive_share(context, "p1") - alive_share(context, "p2"),
        healthy_share(context, "p1") - healthy_share(context, "p2"),
        expected_hidden_knockouts(context),
    ]
