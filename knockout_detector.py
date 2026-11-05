"""Learned detector of the P2 knockouts the records do not report.

A gradient-boosting classifier learns P(faint | change of active
Pokémon) on P1's transitions, whose faints are recorded (the faint row is
skipped, see ``domain.transitions``), and then scores P2's transitions
whose leaver never comes back. No battle outcome is used, so it can be
fitted once, before any feature.

Checked out of fold on P1's transitions (10,212 faints among 38,310
final switches of the training battles, 30 columns, 3 seeds): AUC 0.992,
log loss 0.099, accuracy 96.3%, and at the 0.5 threshold about 93.5%
precision with 92% recall (the earlier hand-written rule: 87% precision,
48% recall). On P2 it repairs 10,130 switches, close to the ~10,100
detectable hidden faints that symmetry with P1's recorded faints predicts.

:class:`DetectorRepair` is the ``TimelineRepair`` of the replay::

    repair = detector_from_files(pokedex, chart, (train,))
    context = BattleReplay(pokedex, repair, Missing()).replay(record)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from data import load_records
from domain.damage import DamageCalculator
from domain.knockouts import CATEGORICAL_FEATURES, DETECTOR_FEATURES
from domain.pokedex import Missing, Pokedex, SpeciesLookup
from domain.records import without_label
from domain.transitions import Transition, TransitionExtractor
from domain.typechart import TypeChart

DEFAULT_THRESHOLD = 0.5
SEED = 42
KEEP_BATTLE = Missing()

Turn = Mapping[str, Any]
Classifier = Any  # scikit-learn style: fit / predict_proba


def default_classifier(
    columns: Sequence[str] = DETECTOR_FEATURES, seed: int = SEED
) -> Classifier:
    """The detector's gradient boosting; the coded columns (``x_status``,
    ``y_code``) are categorical."""
    categorical = [c in CATEGORICAL_FEATURES for c in columns]
    return HistGradientBoostingClassifier(
        max_iter=600,
        learning_rate=0.03,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features=categorical if any(categorical) else None,
        random_state=seed,
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
        self.damage = DamageCalculator(chart)
        self.threshold = threshold
        self._new_classifier = classifier
        self.model: Classifier | None = None

    def fit(self, records: Iterable[Mapping[str, Any]]) -> KnockoutDetector:
        """Learn from P1's final switches (leaver never seen again)."""
        extractor = TransitionExtractor(self.species, self.damage)
        rows, labels = [], []
        for record in records:
            for t in extractor.extract(record, "p1"):
                if t.final and t.features:
                    rows.append(_vector(t))
                    labels.append(int(t.fainted))
        if not rows or len(set(labels)) < 2:
            raise ValueError("no P1 faints and switches to learn from")
        self.model = self._new_classifier().fit(np.array(rows), np.array(labels))
        return self

    def score(self, record: Mapping[str, Any], species: SpeciesLookup) -> dict[int, float]:
        """Turn index -> probability that P2's leaver fainted unseen.

        Only switches whose leaver never comes back and that are not
        recorded faints are scored.
        """
        if self.model is None:
            raise RuntimeError("fit the detector first")
        candidates = [
            t
            for t in TransitionExtractor(species, self.damage).extract(record, "p2")
            if t.final and not t.fainted and t.features
        ]
        if not candidates:
            return {}
        x = np.array([_vector(t) for t in candidates])
        # a handful of rows: extra threads only add overhead
        with threadpool_limits(limits=1):
            p = self.model.predict_proba(x)[:, 1]
        return {t.index: float(v) for t, v in zip(candidates, p, strict=True)}


class DetectorRepair:
    """The fitted detector behind the ``TimelineRepair`` protocol."""

    def __init__(self, detector: KnockoutDetector, pokedex: Pokedex) -> None:
        self._detector = detector
        self._pokedex = pokedex
        self._cache: dict[tuple[Any, ...], dict[int, float]] = {}

    def probabilities(self, record: Mapping[str, Any]) -> dict[int, float]:
        """Scores of the detector, with the stats the battle itself carries.

        Cached by the timeline's content: the repair and the detector
        features ask for the same battles again in every fold.
        """
        key = _fingerprint(record["battle_timeline"])
        cached = self._cache.get(key)
        if cached is None:
            species = self._pokedex.lookup(record, KEEP_BATTLE)
            cached = self._detector.score(record, species)
            self._cache[key] = cached
        return dict(cached)

    def repairs(self, record: Mapping[str, Any]) -> dict[int, str]:
        """Hidden P2 faints: turn index -> name, probability >= threshold."""
        timeline = record["battle_timeline"]
        return {
            i: timeline[i - 1]["p2_pokemon_state"]["name"]
            for i, p in self.probabilities(record).items()
            if p >= self._detector.threshold
        }

    def describe(self) -> dict[str, Any]:
        """Identify this repair in a run; the classifier is refitted, not loaded, for now."""
        return {
            "name": "knockout_detector",
            "version": "refit",
            "params": {
                "threshold": self._detector.threshold,
                "columns": list(DETECTOR_FEATURES),
                "pokedex_sha256": self._pokedex.source_sha256,
            },
        }


def detector_from_files(
    pokedex: Pokedex, chart: TypeChart, paths: Iterable[Path | None]
) -> DetectorRepair:
    """A repair whose detector is fitted on the battles of every given JSONL file."""
    records = [r for path in paths if path is not None for r in load_records(path)]
    detector = KnockoutDetector(pokedex.find, chart)
    return DetectorRepair(detector.fit(without_label(r) for r in records), pokedex)


def _fingerprint(timeline: Sequence[Turn]) -> tuple[Any, ...]:
    """What identifies a timeline: every row's actives, HP, status, moves."""
    return tuple(
        (
            turn["turn"],
            *(
                (
                    turn[f"{side}_pokemon_state"]["name"],
                    turn[f"{side}_pokemon_state"]["hp_pct"],
                    turn[f"{side}_pokemon_state"]["status"],
                    (turn[f"{side}_move_details"] or {}).get("name"),
                )
                for side in ("p1", "p2")
            ),
        )
        for turn in timeline
    )


def _vector(transition: Transition) -> list[float]:
    return [transition.features[name] for name in DETECTOR_FEATURES]
