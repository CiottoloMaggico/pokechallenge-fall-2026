"""Learned detector of the P2 knockouts the records do not report.

A gradient-boosting classifier learns P(faint | change of active
Pokémon) on P1's transitions, whose faints are recorded (the faint row is
skipped, see ``domain.knockouts``), and then scores P2's transitions
whose leaver never comes back. No battle outcome is used, so it can be
fitted on the training and the test battles together, once, before any
feature.

Checked out of fold on P1's transitions (10,212 faints among 38,310
final switches of the training battles): AUC 0.98, and at the 0.5
threshold 89% precision with 88% recall, against 87% precision and 48%
recall of the earlier hand-written rule on the same transitions.

An instance is the ``repair`` of ``BattleContext`` (``data.Repair``)::

    detector = KnockoutDetector(pokedex.find, chart).fit(timelines)
    context = BattleContext.from_record(record, pokedex.get, detector)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from data import load_records
from domain.knockouts import FEATURES, SpeciesLookup, Transition, transitions
from domain.pokedex import Pokedex, PokemonGen2
from domain.typechart import TypeChart

DEFAULT_THRESHOLD = 0.5
SEED = 42

Turn = Mapping[str, Any]
Classifier = Any  # scikit-learn style: fit / predict_proba


def default_classifier() -> Classifier:
    return HistGradientBoostingClassifier(
        max_iter=600,
        learning_rate=0.03,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=SEED,
    )


class KnockoutDetector:
    """P(faint) of P2's switches, learned on P1's recorded faints."""

    def __init__(
        self,
        species: SpeciesLookup,
        chart: TypeChart,
        threshold: float = DEFAULT_THRESHOLD,
        classifier: Callable[[], Classifier] = default_classifier,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        self.species = species
        self.chart = chart
        self.threshold = threshold
        self._new_classifier = classifier
        self.model: Classifier | None = None

    def fit(self, timelines: Iterable[Sequence[Turn]]) -> KnockoutDetector:
        """Learn from P1's final switches (leaver never seen again)."""
        rows, labels = [], []
        for timeline in timelines:
            for t in transitions(timeline, "p1", self.species, self.chart):
                if t.final and t.features:
                    rows.append(_vector(t))
                    labels.append(int(t.fainted))
        if not rows or len(set(labels)) < 2:
            raise ValueError("no P1 faints and switches to learn from")
        self.model = self._new_classifier().fit(np.array(rows), np.array(labels))
        return self

    def probabilities(
        self,
        timeline: Sequence[Turn],
        species: Callable[[str], PokemonGen2] | None = None,
    ) -> dict[int, float]:
        """Turn index -> probability that P2's leaver fainted unseen.

        Only switches whose leaver never comes back and that are not
        recorded faints are scored.
        """
        if self.model is None:
            raise RuntimeError("fit the detector first")
        lookup = _safe(species) if species is not None else self.species
        candidates = [
            t
            for t in transitions(timeline, "p2", lookup, self.chart)
            if t.final and not t.fainted and t.features
        ]
        if not candidates:
            return {}
        x = np.array([_vector(t) for t in candidates])
        p = self.model.predict_proba(x)[:, 1]
        return {t.index: float(v) for t, v in zip(candidates, p, strict=True)}

    def __call__(
        self, timeline: Sequence[Turn], species: Callable[[str], PokemonGen2]
    ) -> dict[int, str]:
        """Hidden P2 faints: turn index -> name, probability >= threshold."""
        p2 = [t["p2_pokemon_state"]["name"] for t in timeline]
        return {
            i: p2[i - 1]
            for i, p in self.probabilities(timeline, species).items()
            if p >= self.threshold
        }


def detector_from_files(
    pokedex: Pokedex, chart: TypeChart, paths: Iterable[Path | None]
) -> KnockoutDetector:
    """A detector fitted on the battles of every given JSONL file."""
    records = [r for path in paths if path is not None for r in load_records(path)]
    detector = KnockoutDetector(pokedex.find, chart)
    return detector.fit(record["battle_timeline"] for record in records)


def _vector(transition: Transition) -> list[float]:
    return [transition.features[name] for name in FEATURES]


def _safe(species: Callable[[str], PokemonGen2]) -> SpeciesLookup:
    """A lookup returning None for unknown names instead of raising."""

    def lookup(name: str) -> PokemonGen2 | None:
        try:
            return species(name)
        except (KeyError, ValueError):
            return None

    return lookup
