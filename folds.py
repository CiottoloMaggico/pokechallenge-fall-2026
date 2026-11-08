"""The folds of an experiment: which rows, checked for leakage, and their feature tables."""

from __future__ import annotations

import pickle
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from artifacts import FeatureMismatchError
from interfaces import FeatureSource, FittableFeatureSource, LabelledBattles, Splitter, is_stateful

FoldData = tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]


class LeakageError(RuntimeError):
    """A split would put the same battle or group in train and validation."""


@dataclass(frozen=True)
class FoldPlan:
    """The fold assignments of an experiment: one k-fold per seed."""

    n_folds: int = 5
    seeds: tuple[int, ...] = (42,)

    def splitters(self) -> Iterator[tuple[int, Splitter]]:
        """``(seed, splitter)``: grouped and stratified, so duplicates stay together."""
        for seed in self.seeds:
            yield seed, StratifiedGroupKFold(self.n_folds, shuffle=True, random_state=seed)


def make_splits(
    splitter: Splitter, y: np.ndarray, groups: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds from ``splitter``, checked for overlapping rows or groups."""
    splits = [
        (np.asarray(tr), np.asarray(va)) for tr, va in splitter.split(np.zeros(len(y)), y, groups)
    ]
    for fold, (train, valid) in enumerate(splits):
        if np.intersect1d(train, valid).size:
            raise LeakageError(f"fold {fold}: rows in both train and validation")
        shared = np.intersect1d(groups[train], groups[valid])
        if shared.size:
            raise LeakageError(
                f"fold {fold}: groups {shared[:5].tolist()} in both train "
                "and validation (use a group-aware splitter)"
            )
    return splits


def read_fold_tables(
    cache: Path, name: str, plan: FoldPlan, with_unlabeled: bool
) -> dict[int, list[FoldData]] | None:
    """Tables written by ``Gym.fold_tables``, or None if a seed is missing.

    Keyed only by the file name: delete them after changing the features or the data.
    Load only files you created (pickle).
    """
    paths = {s: table_path(cache, name, plan.n_folds, s, with_unlabeled) for s in plan.seeds}
    if not all(path.exists() for path in paths.values()):
        return None
    return {seed: pickle.loads(path.read_bytes()) for seed, path in paths.items()}


def table_path(cache: Path, name: str, n_folds: int, seed: int, with_unlabeled: bool) -> Path:
    """File of the tables of one seed."""
    tag = "_unl" if with_unlabeled else ""
    return cache / f"{name}__k{n_folds}_s{seed}{tag}.pkl"


def shared_table(source: FeatureSource, data: LabelledBattles) -> pd.DataFrame | None:
    """The whole table of a stateless source, computed once for every fold; else None."""
    return None if is_stateful(source) else source.to_frame(data.items)


def fold_frames(
    source: FeatureSource,
    shared: pd.DataFrame | None,
    data: LabelledBattles,
    split: tuple[np.ndarray, np.ndarray],
    unlabeled: Sequence[Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and validation features of one fold."""
    train, valid = split
    if shared is not None:
        return shared.iloc[train], shared.iloc[valid]
    return _fold_features(cast(FittableFeatureSource, source), data, split, unlabeled)


def _fold_features(
    source: FittableFeatureSource,
    data: LabelledBattles,
    split: tuple[np.ndarray, np.ndarray],
    unlabeled: Sequence[Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a fresh clone on the training battles only, then transform both.

    Training rows of label-aware features are out of fold (inner folds
    grouped like the outer ones), validation rows use the full fit.
    """
    train, valid = split
    fold_source = source.clone()
    x_train = fold_source.fit_transform(
        [data.items[i] for i in train],
        data.labels[train].tolist(),
        unlabeled,
        data.groups[train].tolist(),
    )
    x_valid = fold_source.to_frame([data.items[i] for i in valid])
    if list(x_train.columns) != list(x_valid.columns):
        raise FeatureMismatchError(
            f"feature set {source.name!r}: train and validation columns differ"
        )
    return x_train, x_valid
