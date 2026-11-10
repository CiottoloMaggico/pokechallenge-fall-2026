"""``StagedFeatureSet`` and ``TableFeature`` with toy features: columns and order, consumed
columns, out-of-fold training rows, unlabelled rows, clone, state, saved runs, the gym."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from artifacts import TrainedRun
from feature_set import FeatureConflictError, FeatureSet, _inner_fold_ids
from folds import FoldPlan
from gym import Gym
from interfaces import Feature, FeatureNotFittedError, LabelledBattles
from models import build
from staged_feature_set import StagedFeatureSet, TableFeature

Battle = Mapping[str, float]


class Raw(Feature[Any]):
    """First stage: ``a`` and the block ``b_0``, ``b_1``."""

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("a", "b_0", "b_1")

    def compute(self, battle: Battle) -> Mapping[str, float]:
        return {"a": battle["a"], "b_0": battle["b"], "b_1": -battle["b"]}


class Centered(TableFeature):
    """Label-free: ``b_0`` minus its mean over the fitted rows; consumes ``b_``."""

    requires_fit = True
    consumes = ("b_",)

    def __init__(self) -> None:
        super().__init__()
        self.mean, self.rows = 0.0, 0

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("b_centered",)

    def reads(self, columns: Sequence[str]) -> list[str]:
        return [c for c in columns if c.startswith("b_")]

    def _fit(self, x: pd.DataFrame, y: np.ndarray | None) -> None:
        self.mean, self.rows = float(x["b_0"].to_numpy().mean()), len(x)

    def _transform(self, x: pd.DataFrame) -> np.ndarray:
        return (x["b_0"].to_numpy() - self.mean)[:, None]


class Target(TableFeature):
    """Label-aware: ``a`` times the win rate and the number of the fitted rows."""

    requires_fit = True
    uses_labels = True

    def __init__(self) -> None:
        super().__init__()
        self.rate, self.rows = 0.0, 0

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("a_target", "target_rows")

    def reads(self, columns: Sequence[str]) -> list[str]:
        return ["a"]

    def _fit(self, x: pd.DataFrame, y: np.ndarray | None) -> None:
        assert y is not None
        self.rate, self.rows = float(y.mean()), len(x)

    def _transform(self, x: pd.DataFrame) -> np.ndarray:
        a = x["a"].to_numpy()
        return np.column_stack([a * self.rate, np.full(len(a), self.rows)])


def staged() -> StagedFeatureSet[Any]:
    base = FeatureSet[Any]("raw")
    base.register(Raw())
    return StagedFeatureSet[Any]("toy", base, [Centered(), Target()])


def battles(n: int = 60, seed: int = 0) -> tuple[list[dict[str, float]], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    items = [{"a": float(rng.normal()), "b": float(rng.normal())} for _ in range(n)]
    labels = [int(item["a"] + rng.normal() > 0) for item in items]
    groups = [i // 2 for i in range(n)]  # pairs of duplicates
    return items, labels, groups


def test_columns_order_and_consumed_columns() -> None:
    items, labels, groups = battles()
    features = staged()
    x = features.fit_transform(items, labels, (), groups)
    assert list(x.columns) == ["a", "b_centered", "a_target", "target_rows"]
    assert features.declared_columns == list(x.columns)
    assert features.feature_ids == ["Raw", "Centered", "Target"]
    assert features.uses_labels and features.is_stateful and not features.requires_fit
    assert list(features.to_frame(items[:5]).columns) == list(x.columns)


def test_training_rows_come_from_a_copy_fitted_without_their_inner_fold() -> None:
    items, labels, groups = battles()
    features = staged()
    x = features.fit_transform(items, labels, (), groups)
    folds = _inner_fold_ids(len(items), groups, 5, 0)
    table = features.base.to_frame(items)
    target = features.stage[1]
    for fold in range(5):
        inside = [i for i, f in enumerate(folds) if f == fold]
        outside = [i for i, f in enumerate(folds) if f != fold]
        twin = copy.deepcopy(target)
        twin.reset()
        twin.fit(table.iloc[outside], [labels[i] for i in outside])
        expected = twin.transform(table.iloc[inside])
        assert np.array_equal(x.loc[inside, ["a_target", "target_rows"]].to_numpy(), expected)
    # the label-free column and the state used later come from all the training rows
    assert target.rows == len(items)  # type: ignore[attr-defined]
    assert np.allclose(x["b_centered"], table["b_0"] - table["b_0"].mean())


def test_label_aware_features_never_see_unlabelled_rows() -> None:
    items, labels, groups = battles()
    extra, _, _ = battles(20, seed=1)
    features = staged()
    features.fit_transform(items, labels, extra, groups)
    centered, target = features.stage
    assert centered.rows == len(items) + len(extra)  # type: ignore[attr-defined]
    assert target.rows == len(items)  # type: ignore[attr-defined]
    features.clone().fit(items, labels, extra)
    with pytest.raises(ValueError, match="needs the training labels"):
        staged().fit(items, None)


def test_clone_is_unfitted_and_the_state_round_trips() -> None:
    items, labels, groups = battles()
    features = staged()
    with pytest.raises(FeatureNotFittedError):
        features.to_frame(items)
    features.fit_transform(items, labels, (), groups)
    twin = features.clone()
    assert twin.requires_fit
    twin.set_state(features.get_state())
    assert twin.to_frame(items).equals(features.to_frame(items))


def test_conflicting_columns_are_refused() -> None:
    class Clash(Target):
        @property
        def output_names(self) -> tuple[str, ...]:
            return ("a", "target_rows")

    base = FeatureSet[Any]("raw")
    base.register(Raw())
    with pytest.raises(FeatureConflictError):
        StagedFeatureSet[Any]("clash", base, [Clash()]).declared_columns  # noqa: B018


def test_a_saved_run_predicts_and_the_gym_evaluates(tmp_path: Path) -> None:
    items, labels, groups = battles(100)
    data = LabelledBattles(items, np.array(labels), np.arange(100), np.array(groups))
    (run,) = Gym([build("logreg")], [staged()]).train(data)
    run.save(tmp_path / "run")
    loaded = TrainedRun.load(tmp_path / "run", staged())
    assert np.array_equal(loaded.predict_proba(items[:10]), run.predict_proba(items[:10]))
    result = Gym([build("logreg")], [staged()], FoldPlan(5, (0,))).evaluate(data)
    assert len(result.folds) == 5 and set(result.folds["feature_set"]) == {"toy"}
