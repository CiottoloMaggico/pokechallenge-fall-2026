"""The hidden P2 knockouts of v11's detector artifact (``python main.py detector train-v11``).

An artifact folder (``results/addons/knockouts_v11/<version>/``) holds the ``TrainedRun`` of the
detector fitted on all P1 rows of the training file (``lgbm_ko_v11`` on ``ko_v11``),
``tables.pkl`` with the cross-fitted probabilities of the training battles and ``info.json``
(version, settings, hashes, diagnostics). Loading it refits nothing::

    folder = Path("results/addons/knockouts_v11/v1")
    repair = V11KnockoutRepair.from_artifact(folder, pokedex, chart)

``probabilities`` has the shape of the original's ``battle.p2_faint``: for every P2 switch of
``domain.switches.SwitchExtractor``, in order, 1.0 for a recorded faint, P(faint) for a
candidate, 0.0 for the others, where P(faint) is

* for a training battle of the artifact, its cross-fitted value (the mean over the seeds of the
  out-of-fold probabilities of the standard ``FoldPlan``), keyed by battle id and
  ``fingerprint`` (training duplicates have distinct ids, test ids may repeat training ones);
* for a training battle truncated at a cut of v11's forecast, the cross-fitted value of the
  detector of that cut, same key on the truncated record;
* for any other record, the full detector's (one batch per record).
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from threadpoolctl import ThreadpoolController

from addons.knockouts_model import ArtifactError
from artifacts import TrainedRun
from domain.pokedex import Missing, Pokedex
from domain.records import fingerprint
from domain.switches import SwitchExtractor
from domain.typechart import TypeChart
from features.transitions import ko_v11

INFO_FILE = "info.json"
TABLES_FILE = "tables.pkl"
THRESHOLD = 0.5
DESCRIBED = ("feature_set", "model", "params", "cuts", "seeds", "train_sha256", "pokedex_sha256")
KEEP_BATTLE = Missing()

Key = tuple[Any, str]
Probabilities = dict[int, float]
Tables = dict[str, Any]  # "full": {key: probabilities}, "cuts": {cut: {key: probabilities}}


def battle_key(record: Mapping[str, Any]) -> Key:
    """(battle id, content hash): what identifies a battle of the artifact's tables."""
    return record["battle_id"], fingerprint(record)


def load_artifact(folder: Path) -> tuple[TrainedRun, Tables, dict[str, Any]]:
    """The full detector, the cross-fitted tables and the ``info.json`` of an artifact folder."""
    if not (folder / INFO_FILE).exists():
        raise ArtifactError(
            f"no v11 detector artifact in {folder}: "
            f"train it with `python main.py detector train-v11 --version {folder.name}`"
        )
    info = json.loads((folder / INFO_FILE).read_text())
    tables = pickle.loads((folder / TABLES_FILE).read_bytes())
    return TrainedRun.load(folder, ko_v11.build()), tables, info


class V11KnockoutRepair:
    """``TimelineRepair`` with v11's detector: P(faint) of P2's switches (see the module)."""

    def __init__(
        self,
        run: TrainedRun,
        tables: Tables,
        info: Mapping[str, Any],
        pokedex: Pokedex,
        chart: TypeChart,
    ) -> None:
        self._run = run
        self._full: dict[Key, Probabilities] = tables["full"]
        self._cuts: dict[int, dict[Key, Probabilities]] = tables["cuts"]
        self._info = dict(info)
        self._pokedex = pokedex
        self._chart = chart
        self._cache: dict[str, Probabilities] = {}
        # found once, with the model's libraries loaded: a new lookup costs more than a battle
        self._threads = ThreadpoolController()

    @classmethod
    def from_artifact(cls, folder: Path, pokedex: Pokedex, chart: TypeChart) -> V11KnockoutRepair:
        """Load the artifact in ``folder``, written by ``detector train-v11``."""
        return cls(*load_artifact(folder), pokedex, chart)

    def probabilities(self, record: Mapping[str, Any]) -> Probabilities:
        """Timeline index of every P2 switch -> P(the P2 Pokémon that left there fainted)."""
        key = battle_key(record)
        found = self._full.get(key)
        if found is None:
            found = self._cuts.get(len(record["battle_timeline"]), {}).get(key)
        if found is None:
            found = self._cache.get(key[1])
        if found is None:
            found = self._cache[key[1]] = self._score(record)
        return dict(found)

    def repairs(self, record: Mapping[str, Any]) -> dict[int, str]:
        """Timeline index -> name of the P2 Pokémon that left there, probability >= 0.5."""
        timeline = record["battle_timeline"]
        return {
            i: timeline[i - 1]["p2_pokemon_state"]["name"]
            for i, p in self.probabilities(record).items()
            if p >= THRESHOLD
        }

    def describe(self) -> dict[str, Any]:
        """Version and settings of the artifact, from its ``info.json``."""
        return {
            "name": "knockouts_v11",
            "version": self._info["version"],
            "params": {key: self._info[key] for key in DESCRIBED},
        }

    def _score(self, record: Mapping[str, Any]) -> Probabilities:
        """The full detector on the record's candidates; recorded faints 1.0, others 0.0."""
        species = self._pokedex.lookup(record, KEEP_BATTLE)
        switches = SwitchExtractor(species, self._chart).extract(record, "p2")
        candidates = [s for s in switches if s.final and not s.recorded]
        scored: Probabilities = {}
        if candidates:
            # a handful of rows: extra threads only add overhead
            with self._threads.limit(limits=1):
                p = self._run.predict_proba(candidates)
            scored = {s.index: float(v) for s, v in zip(candidates, p, strict=True)}
        return {s.index: 1.0 if s.recorded else scored.get(s.index, 0.0) for s in switches}
