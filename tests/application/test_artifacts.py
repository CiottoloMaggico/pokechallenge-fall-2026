"""A trained run saved and loaded again: same predictions, repair manifest included."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

import model_catalog
from artifacts import RUN_INFO_FILE, FeatureMismatchError, TrainedRun, save_runs
from data import Dataset
from domain.pokedex import PokedexBuilder
from domain.records import read_records
from domain.repair import NO_REPAIR
from feature_catalog import build
from feature_set import FeatureSet
from gym import Gym
from interfaces import Feature

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "battles.jsonl"
MANIFEST = NO_REPAIR.describe()


class HalfMean(Feature[float]):
    """The training mean of the battles, half of it for each battle: learns state, no labels."""

    requires_fit = True

    def __init__(self) -> None:
        super().__init__("half_mean")
        self.mean = 0.0

    def _fit(self, battles: Sequence[float], labels: Sequence[int] | None) -> None:
        self.mean = float(np.mean(battles))

    def compute(self, battle: float) -> Mapping[str, float]:
        return {"half_mean": battle - self.mean / 2}


def source(name: str = "toy") -> FeatureSet[float]:
    features = FeatureSet[float](name)
    features.register(HalfMean())
    return features


@pytest.fixture
def run() -> TrainedRun:
    battles = [float(i) for i in range(40)]
    labels = [i % 2 for i in range(40)]
    gym = Gym(model_catalog.build("logreg"), source())
    return gym.train(battles, labels, repair=MANIFEST)[0]


def test_the_manifest_of_the_repair_is_saved(run: TrainedRun, tmp_path: Path) -> None:
    folder = run.save(tmp_path / "run")
    assert json.loads((folder / RUN_INFO_FILE).read_text())["repair"] == MANIFEST
    assert TrainedRun.load(folder, source()).repair == MANIFEST


def test_a_run_saved_before_the_manifest_loads_without_it(run: TrainedRun, tmp_path: Path) -> None:
    folder = run.save(tmp_path / "run")
    info = json.loads((folder / RUN_INFO_FILE).read_text())
    del info["repair"]
    (folder / RUN_INFO_FILE).write_text(json.dumps(info))
    assert TrainedRun.load(folder, source()).repair is None


def test_a_loaded_run_predicts_the_same(run: TrainedRun, tmp_path: Path) -> None:
    folder = save_runs([run], tmp_path)[0]
    assert folder.name == "toy__logreg"
    battles = [1.0, 2.5, 30.0]
    loaded = TrainedRun.load(folder, source(), sample_battles=battles)
    assert np.array_equal(loaded.predict_proba(battles), run.predict_proba(battles))


def test_a_run_refuses_another_feature_set(run: TrainedRun, tmp_path: Path) -> None:
    folder = run.save(tmp_path / "run")
    with pytest.raises(FeatureMismatchError, match="trained on feature set"):
        TrainedRun.load(folder, source("other"))


def test_a_run_refuses_changed_columns(run: TrainedRun) -> None:
    changed = TrainedRun(run.source, run.model, ("another_column",))
    with pytest.raises(FeatureMismatchError, match="changed since fit"):
        changed.check_columns([1.0])


def test_the_main_feature_set_round_trips(tmp_path: Path) -> None:
    pokedex = PokedexBuilder().add_records(read_records(FIXTURES)).build()
    data = Dataset.from_jsonl(FIXTURES, pokedex)
    gym = Gym(model_catalog.build("logreg"), build("compact_v5"))
    run = gym.train(data.contexts, data.labels, groups=data.duplicate_groups(), repair=MANIFEST)[0]
    loaded = TrainedRun.load(run.save(tmp_path / "run"), build("compact_v5"), data.contexts[:5])
    assert loaded.columns == run.columns
    assert np.array_equal(
        loaded.predict_proba(data.contexts[:20]), run.predict_proba(data.contexts[:20])
    )
