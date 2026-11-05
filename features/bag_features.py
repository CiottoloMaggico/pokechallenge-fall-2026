"""Who is on the field and what they showed, compressed into one score.

``bag_logit``: a logistic regression (L2, standardised inputs) on three
"bags" over the species and moves common in the training battles, each
entry P1's value minus P2's:

* species presence (P1's team, P2's revealed Pokémon);
* HP left of each species (0 if fainted or absent);
* moves revealed by any Pokémon of the team.

Its decision value goes through ``tanh(z / 2)`` (= 2 P(P1 wins) - 1), so
the feature lies in [-1, 1] and is positive when the line-ups favour P1.
It uses the labels: the gym fits it inside each training fold and gives
the training rows out-of-fold values.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from domain.battle import BattleContext, PlayerState
from features.turn_features import check_turn
from interfaces import Feature

DEFAULT_C = 0.01
MIN_SPECIES = 20
MIN_MOVES = 30


class BagLogit(Feature[BattleContext]):
    """``bag_logit`` at the sampled turn (see the module docstring)."""

    requires_fit = True
    uses_labels = True

    def __init__(
        self,
        turn: int | None = None,
        c: float = DEFAULT_C,
        min_species: int = MIN_SPECIES,
        min_moves: int = MIN_MOVES,
    ) -> None:
        super().__init__("bag_logit")
        if c <= 0:
            raise ValueError("c must be positive")
        self.turn = check_turn(turn)
        self.c = c
        self.min_species = min_species
        self.min_moves = min_moves
        self.species: list[str] = []
        self.moves: list[str] = []
        self.model: Pipeline | None = None

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("bag_logit",)

    def _fit(
        self, battles: Sequence[BattleContext], labels: Sequence[int] | None
    ) -> None:
        if labels is None:
            raise ValueError("the bag logit needs the training labels")
        contexts = [battle.at_turn(self.turn) for battle in battles]
        species: Counter[str] = Counter()
        moves: Counter[str] = Counter()
        for context in contexts:
            for side in ("p1", "p2"):
                team = context.side(side).team
                species.update(list(team))
                moves.update(_revealed_moves(context.side(side)))
        self.species = sorted(s for s, n in species.items() if n >= self.min_species)
        self.moves = sorted(m for m, n in moves.items() if n >= self.min_moves)
        x = np.array([self._vector(context) for context in contexts])
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=self.c, max_iter=5000)
        )
        self.model = model.fit(x, np.asarray(labels))

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        if self.model is None:
            raise RuntimeError("fit the feature first")
        context = battle.at_turn(self.turn)
        z = float(self.model.decision_function(np.array([self._vector(context)]))[0])
        return {"bag_logit": math.tanh(z / 2)}

    def _vector(self, context: BattleContext) -> list[float]:
        p1, p2 = context.side("p1"), context.side("p2")
        hp1, hp2 = _hp_left(p1), _hp_left(p2)
        moves1, moves2 = _revealed_moves(p1), _revealed_moves(p2)
        presence = [float(s in p1.team) - float(s in p2.team) for s in self.species]
        health = [hp1.get(s, 0.0) - hp2.get(s, 0.0) for s in self.species]
        shown = [float(m in moves1) - float(m in moves2) for m in self.moves]
        return [*presence, *health, *shown]


def _hp_left(player: PlayerState) -> dict[str, float]:
    return {
        name: 0.0 if pokemon.fainted else pokemon.hp_pct
        for name, pokemon in player.team.items()
    }


def _revealed_moves(player: PlayerState) -> set[str]:
    return {move for pokemon in player.team.values() for move in pokemon.known_moves}
