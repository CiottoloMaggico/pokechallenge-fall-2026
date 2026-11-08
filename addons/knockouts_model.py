"""The hidden P2 knockouts found by a detector artifact (``python main.py detector train``).

An artifact folder (``results/addons/knockouts/<version>/``) holds a ``TrainedRun`` fitted on
P1's switches of the training file and ``info.json`` (version, feature set, model, threshold,
hashes, evaluation). Loading it refits nothing, so the repair of a battle is the same in every
command and in every process::

    repair = ModelKnockoutRepair.from_artifact(Path("results/addons/knockouts/v3"), pokedex, damage)
    context = BattleReplay(pokedex, repair, Missing()).replay(record)

Scores are cached by ``fingerprint`` of the record (truncated copies included: the key only
has to identify the content), because the detector reads more than the timeline: the species
whose stats only the battle carries (``Pokedex.lookup``) come from the team details.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from threadpoolctl import ThreadpoolController

from artifacts import TrainedRun
from domain.damage import DamageCalculator
from domain.pokedex import Missing, Pokedex
from domain.records import fingerprint
from domain.transitions import TransitionExtractor
from features import transitions

INFO_FILE = "info.json"
DESCRIBED = ("feature_set", "model", "params", "threshold", "train_sha256", "pokedex_sha256")
KEEP_BATTLE = Missing()


class ArtifactError(Exception):
    """A detector artifact is missing."""


def load_artifact(folder: Path) -> tuple[TrainedRun, dict[str, Any]]:
    """The fitted detector and the ``info.json`` of an artifact folder."""
    if not (folder / INFO_FILE).exists():
        raise ArtifactError(
            f"no detector artifact in {folder}: "
            f"train it with `python main.py detector train --version {folder.name}`"
        )
    info = json.loads((folder / INFO_FILE).read_text())
    features = transitions.build(TrainedRun.saved_feature_set(folder))
    return TrainedRun.load(folder, features), info


class ModelKnockoutRepair:
    """``TimelineRepair`` with a fitted detector: P(faint) of P2's candidate switches."""

    def __init__(
        self,
        run: TrainedRun,
        info: Mapping[str, Any],
        pokedex: Pokedex,
        damage: DamageCalculator,
    ) -> None:
        self._run = run
        self._info = dict(info)
        self._threshold = float(info["threshold"])
        self._pokedex = pokedex
        self._damage = damage
        self._cache: dict[str, dict[int, float]] = {}
        # found once, with the model's libraries loaded: a new lookup costs more than a battle
        self._threads = ThreadpoolController()

    @classmethod
    def from_artifact(
        cls, folder: Path, pokedex: Pokedex, damage: DamageCalculator
    ) -> ModelKnockoutRepair:
        """Load the artifact in ``folder``, written by ``detector train``."""
        return cls(*load_artifact(folder), pokedex, damage)

    def probabilities(self, record: Mapping[str, Any]) -> dict[int, float]:
        """Turn index -> P(the P2 Pokémon that left at that row fainted unseen)."""
        key = fingerprint(record)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._score(record)
            self._cache[key] = cached
        return dict(cached)

    def repairs(self, record: Mapping[str, Any]) -> dict[int, str]:
        """Hidden P2 faints: turn index -> name, probability >= threshold."""
        timeline = record["battle_timeline"]
        return {
            i: timeline[i - 1]["p2_pokemon_state"]["name"]
            for i, p in self.probabilities(record).items()
            if p >= self._threshold
        }

    def describe(self) -> dict[str, Any]:
        """Version and settings of the artifact, from its ``info.json``."""
        return {
            "name": "knockouts_model",
            "version": self._info["version"],
            "params": {key: self._info[key] for key in DESCRIBED},
        }

    def _score(self, record: Mapping[str, Any]) -> dict[int, float]:
        species = self._pokedex.lookup(record, KEEP_BATTLE)
        candidates = TransitionExtractor(species, self._damage).candidates(record)
        if not candidates:
            return {}
        # a handful of rows: extra threads only add overhead
        with self._threads.limit(limits=1):
            p = self._run.predict_proba(candidates)
        return {t.index: float(v) for t, v in zip(candidates, p, strict=True)}
