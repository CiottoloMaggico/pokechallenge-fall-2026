"""The gym on toy rows: folds, leakage checks, seeds, parallel seeds, fold tables, runs."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from feature_set import FeatureSet
from folds import FoldPlan, LeakageError, make_splits, read_fold_tables
from gym import Gym, GymResult
from interfaces import Feature, FitContext, LabelledBattles, Model

N = 60
PLAN = FoldPlan(3, (1, 2))
CLEAN_PROCESS = "GYM_TEST_CLEAN_PROCESS"


class Noisy(Feature[float]):
    """A stateless column."""

    def __init__(self) -> None:
        super().__init__("noisy")

    def compute(self, battle: float) -> Mapping[str, float]:
        return {"noisy": battle}


class SeenMean(Feature[float]):
    """Remembers the battles of its fit, to show what a fold was fitted on."""

    requires_fit = True

    def __init__(self) -> None:
        super().__init__("seen_mean")
        self.mean = 0.0

    def _fit(self, battles: Sequence[float], labels: Sequence[int] | None) -> None:
        self.mean = float(np.mean(battles))

    def compute(self, battle: float) -> Mapping[str, float]:
        return {"seen_mean": self.mean}


class MeanModel(Model):
    """Predicts the training mean: numpy only, so it is safe in forked workers."""

    def __init__(self, name: str = "mean") -> None:
        super().__init__(name)
        self.mean = 0.5

    def clone(self) -> MeanModel:
        return MeanModel(self.name)

    def get_params(self) -> dict[str, Any]:
        return {}

    def _fit(self, x: pd.DataFrame, y: np.ndarray) -> None:
        self.mean = float(y.mean())

    def _predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(len(x), self.mean)


def stateless(name: str = "plain") -> FeatureSet[float]:
    features = FeatureSet[float](name)
    features.register(Noisy())
    return features


def stateful() -> FeatureSet[float]:
    features = FeatureSet[float]("fitted")
    features.register(SeenMean())
    return features


@pytest.fixture
def data() -> LabelledBattles:
    """Rows 2k and 2k + 1 are one duplicated battle; labels are balanced inside the folds."""
    ids = np.arange(N)
    return LabelledBattles([float(i) for i in ids], (ids // 2) % 2, ids * 10, ids // 2)


def test_the_default_plan_is_one_grouped_five_fold() -> None:
    (seed, splitter), *others = FoldPlan().splitters()
    assert (seed, others) == (42, [])
    assert splitter.get_n_splits() == 5  # type: ignore[attr-defined]


def test_data_of_different_lengths_is_refused() -> None:
    with pytest.raises(ValueError, match="equal length"):
        LabelledBattles([1.0], np.zeros(2), np.zeros(2), np.zeros(2))


def test_the_folds_keep_groups_together_and_use_every_row(data: LabelledBattles) -> None:
    for _, splitter in PLAN.splitters():
        splits = make_splits(splitter, data.labels, data.groups)
        assert sorted(np.concatenate([valid for _, valid in splits])) == list(range(N))
        for train, valid in splits:
            assert not set(data.groups[train]) & set(data.groups[valid])


class FixedSplit:
    """A splitter that returns the folds it is told to."""

    def __init__(self, train: list[int], valid: list[int]) -> None:
        self.fold = (np.array(train), np.array(valid))

    def split(self, x: Any, y: Any, groups: Any, /) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        return [self.fold]


def test_a_split_that_shares_a_row_is_refused() -> None:
    with pytest.raises(LeakageError, match="rows in both"):
        make_splits(FixedSplit([0, 1], [1, 2]), np.zeros(3), np.arange(3))


def test_a_split_that_shares_a_group_is_refused() -> None:
    with pytest.raises(LeakageError, match="groups"):
        make_splits(FixedSplit([0], [1]), np.zeros(2), np.zeros(2))


def test_evaluate_has_every_combination_of_every_seed_and_fold(data: LabelledBattles) -> None:
    gym = Gym([MeanModel("a"), MeanModel("b")], [stateless("x"), stateless("y")], PLAN)
    result = gym.evaluate(data)
    assert len(result.folds) == 2 * 2 * 2 * 3
    assert set(result.folds["seed"]) == {1, 2}
    assert result.meta["seeds"] == [1, 2]
    one = result.predictions.query("feature_set == 'x' and model == 'a' and seed == 1")
    assert sorted(one["id"]) == sorted(data.ids)
    assert list(result.summary()["n_seeds"]) == [2, 2, 2, 2]
    assert list(result.by_seed().columns) == ["feature_set", "model", "seed 1", "seed 2"]


def test_one_seed_has_a_seed_column_and_the_short_summary(data: LabelledBattles) -> None:
    result = Gym([MeanModel()], [stateless()], FoldPlan(3, (7,))).evaluate(data)
    assert set(result.folds["seed"]) == {7}
    assert "n_seeds" not in result.summary()


def test_a_stateful_set_is_fitted_on_the_training_part_only(data: LabelledBattles) -> None:
    gym = Gym([MeanModel()], [stateful()], FoldPlan(3, (1,)))
    tables = gym.fold_tables(data, stateful(), FoldPlan(3, (1,)))[1]
    splits = make_splits(next(FoldPlan(3, (1,)).splitters())[1], data.labels, data.groups)
    for (train, _), (_, _, x_valid, _) in zip(splits, tables, strict=True):
        expected = np.mean([data.items[i] for i in train])
        assert x_valid["seen_mean"].tolist() == [expected] * len(x_valid)


def test_unlabeled_battles_reach_label_free_features(data: LabelledBattles) -> None:
    plan = FoldPlan(3, (1,))
    alone = Gym.fold_tables(data, stateful(), plan)[1][0][2]["seen_mean"].iloc[0]
    fed = Gym.fold_tables(data, stateful(), plan, unlabeled=[1000.0])[1][0][2]["seen_mean"].iloc[0]
    assert fed > alone


def test_seeds_in_parallel_give_the_same_tables(data: LabelledBattles) -> None:
    """The gym forks its workers. Earlier tests load LightGBM, XGBoost or CatBoost in the test
    process, after which a fork can hang (AGENTS.md, rule 6), so the comparison reruns this
    test alone in a fresh interpreter."""
    if os.environ.get(CLEAN_PROCESS) != "1":
        node = f"{__file__}::test_seeds_in_parallel_give_the_same_tables"
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", node]
        env = {**os.environ, CLEAN_PROCESS: "1"}
        done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=240)
        assert done.returncode == 0, done.stdout[-3000:]
        return
    sequence = Gym([MeanModel()], [stateless()], PLAN).evaluate(data)
    parallel = Gym([MeanModel()], [stateless()], PLAN, jobs=2).evaluate(data)
    pd.testing.assert_frame_equal(sequence.predictions, parallel.predictions)
    pd.testing.assert_frame_equal(
        sequence.folds.drop(columns="fit_seconds"), parallel.folds.drop(columns="fit_seconds")
    )


def test_fold_tables_match_what_evaluate_fits_on(data: LabelledBattles) -> None:
    result = Gym([MeanModel()], [stateless()], PLAN).evaluate(data)
    tables = Gym.fold_tables(data, stateless(), PLAN)
    assert set(tables) == {1, 2}
    frame, predictions = result.folds, result.predictions
    for seed, folds in tables.items():
        for fold, (x_train, y_train, x_valid, y_valid) in enumerate(folds):
            row = frame[(frame["seed"] == seed) & (frame["fold"] == fold)].iloc[0]
            assert (row["n_train"], row["n_valid"]) == (len(y_train), len(y_valid))
            assert len(x_train) == len(y_train) and len(x_valid) == len(y_valid)
            preds = predictions[(predictions["seed"] == seed) & (predictions["fold"] == fold)]
            assert preds["proba"].tolist() == pytest.approx([y_train.mean()] * len(y_valid))


def test_cached_fold_tables_are_read_back_only_when_every_seed_is_there(
    data: LabelledBattles, tmp_path: Path
) -> None:
    source = stateless()
    assert read_fold_tables(tmp_path, source.name, PLAN, False) is None
    built = Gym.fold_tables(data, source, PLAN, cache=tmp_path)
    assert (tmp_path / "plain__k3_s1.pkl").exists()
    read = read_fold_tables(tmp_path, source.name, PLAN, False)
    assert read is not None
    pd.testing.assert_frame_equal(read[2][1][0], built[2][1][0])
    assert read_fold_tables(tmp_path, source.name, FoldPlan(3, (1, 9)), False) is None
    assert read_fold_tables(tmp_path, source.name, PLAN, True) is None


def test_progress_counts_the_folds_of_every_seed(data: LabelledBattles) -> None:
    seen: list[tuple[int, int]] = []
    Gym.fold_tables(data, stateless(), PLAN, progress=lambda d, t, _: seen.append((d, t)))
    assert seen == [(k, 6) for k in range(1, 7)]


def test_probes_get_every_fit(data: LabelledBattles) -> None:
    def sizes(context: FitContext) -> dict[str, int]:
        return {"n": len(context.y_valid)}

    result = Gym([MeanModel()], [stateless()], PLAN, probes={"sizes": sizes}).evaluate(data)
    assert len(result.probes["sizes"]) == 6
    assert sum(result.probes["sizes"]["n"]) == 2 * N


def test_train_fits_a_clone_on_everything(data: LabelledBattles) -> None:
    source = stateful()
    run = Gym([MeanModel()], [source]).train(data, repair={"name": "none"})[0]
    assert run.repair == {"name": "none"}
    assert run.columns == ("seen_mean",)
    assert run.source is not source
    assert run.predict_proba([1.0]).tolist() == [float(data.labels.mean())]


def test_results_survive_an_export(data: LabelledBattles, tmp_path: Path) -> None:
    result = Gym([MeanModel()], [stateless()], PLAN).evaluate(data)
    loaded = GymResult.load(result.export(tmp_path / "out"))
    assert loaded.meta["seeds"] == [1, 2]
    pd.testing.assert_frame_equal(loaded.summary(), result.summary())


def test_repeated_names_are_refused() -> None:
    with pytest.raises(ValueError, match="model names"):
        Gym([MeanModel(), MeanModel()], [stateless()])
    with pytest.raises(ValueError, match="feature set names"):
        Gym([MeanModel()], [stateless(), stateless()])
