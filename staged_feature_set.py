"""A feature set in two stages: battle features, then features learned on their table.

Some features are transformations of columns that other features already compute (a logistic
regression of a block of columns, a spline expansion, a team effect against a base logit). As
ordinary ``Feature``s each would recompute its inputs from the battles, and every inner-fold copy
of a label-aware one would recompute them again. A :class:`StagedFeatureSet` computes the battle
features once (its first stage, a :class:`feature_set.FeatureSet`), then runs
:class:`TableFeature`s on that table: only the cheap learned part is refitted in the inner folds.

The leakage rules of ``feature_set.py`` hold for both stages: label-aware table features are
fitted on the labelled training rows only and give their training rows out of fold, with the
same grouped inner folds as ``FeatureSet.fit_transform``; label-free ones may also learn from the
rows of ``unlabeled`` battles. A table feature may *consume* first-stage columns (by prefix):
they are its inputs, not outputs of the set.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from feature_set import DEFAULT_INNER_FOLDS, FeatureConflictError, FeatureSet, _inner_fold_ids
from interfaces import FeatureNotFittedError


class TableFeature(ABC):
    """A feature computed from the first stage's table (one output row per table row).

    Subclasses declare ``output_names``, the columns they ``reads`` and, if fitted, implement
    ``_fit`` (``requires_fit = True``; ``uses_labels = True`` if it needs the labels). ``consumes``
    lists the prefixes of first-stage columns the set drops from its output. ``inner_oof = False``
    if the feature never gives in-sample values on its own training rows. The state is saved like
    a ``Feature``'s (every instance attribute, so keep it picklable).
    """

    requires_fit: bool = False
    uses_labels: bool = False
    inner_oof: bool = True
    consumes: tuple[str, ...] = ()

    def __init__(self, feature_id: str | None = None) -> None:
        self._feature_id = feature_id or type(self).__name__
        self._fitted = False

    @property
    def feature_id(self) -> str:
        """Registry id (defaults to the class name)."""
        return self._feature_id

    @property
    @abstractmethod
    def output_names(self) -> tuple[str, ...]:
        """The produced columns, in order."""

    @abstractmethod
    def reads(self, columns: Sequence[str]) -> list[str]:
        """The columns of the table it uses, in the order it uses them."""

    @property
    def is_fitted(self) -> bool:
        """True when the feature can transform: fitted, or nothing to fit."""
        return self._fitted or not self.requires_fit

    def fit(self, table: pd.DataFrame, labels: Sequence[int] | None = None) -> TableFeature:
        """Learn from the rows of ``table`` (labels if and only if ``uses_labels``)."""
        if self.uses_labels:
            if labels is None or len(labels) != len(table):
                raise ValueError(f"feature {self.feature_id!r} needs one label per row")
        elif labels is not None:
            raise ValueError(f"feature {self.feature_id!r} does not use labels: do not pass them")
        y = None if labels is None else np.asarray(labels)
        self._fit(table.loc[:, self.reads(list(table.columns))], y)
        self._fitted = True
        return self

    def _fit(self, x: pd.DataFrame, y: np.ndarray | None) -> None:  # noqa: B027
        """Hook for fitted features: ``x`` holds the columns of :meth:`reads`."""

    def transform(self, table: pd.DataFrame) -> np.ndarray:
        """One row of ``output_names`` per row of ``table``."""
        if not self.is_fitted:
            raise FeatureNotFittedError(f"feature {self.feature_id!r} needs fit() before use")
        out = np.asarray(self._transform(table.loc[:, self.reads(list(table.columns))]), float)
        if out.shape != (len(table), len(self.output_names)):
            raise ValueError(f"feature {self.feature_id!r}: output of shape {out.shape}")
        return out

    @abstractmethod
    def _transform(self, x: pd.DataFrame) -> np.ndarray:
        """The output rows for the columns of :meth:`reads`."""

    def reset(self) -> None:
        """Forget the fitted flag (used on copies before refitting)."""
        self._fitted = False

    def get_state(self) -> dict[str, Any]:
        """Learned state (default: every attribute)."""
        return copy.deepcopy(vars(self))

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore a state returned by :meth:`get_state`."""
        vars(self).update(copy.deepcopy(state))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.feature_id!r})"


class StagedFeatureSet[BattleT]:
    """``base`` (battle features), then the table features of ``stage``, in order.

    Output: the columns of ``base`` but the consumed ones (order kept), then the outputs of the
    table features. Implements ``interfaces.FittableFeatureSource`` like ``FeatureSet``.
    """

    def __init__(self, name: str, base: FeatureSet[BattleT], stage: Sequence[TableFeature]) -> None:
        if not name:
            raise ValueError("a feature set needs a non-empty name")
        ids = [*base.feature_ids, *(f.feature_id for f in stage)]
        if len(set(ids)) != len(ids):
            raise FeatureConflictError(f"{name}: repeated feature ids {ids}")
        self._name, self.base, self.stage = name, base, list(stage)

    @property
    def name(self) -> str:
        """Name of the set, used to tag results and saved models."""
        return self._name

    @property
    def feature_ids(self) -> list[str]:
        """Ids of the first stage, then of the table features."""
        return [*self.base.feature_ids, *(f.feature_id for f in self.stage)]

    @property
    def uses_labels(self) -> bool:
        return self.base.uses_labels or any(f.uses_labels for f in self.stage)

    @property
    def requires_fit(self) -> bool:
        return self.base.requires_fit or any(not f.is_fitted for f in self.stage)

    @property
    def is_stateful(self) -> bool:
        return self.base.is_stateful or any(f.requires_fit for f in self.stage)

    @property
    def declared_columns(self) -> list[str] | None:
        """All output columns, if the first stage declares its own; else None."""
        columns = self.base.declared_columns
        return None if columns is None else self._output_names(columns)

    def _consumed(self) -> tuple[str, ...]:
        return tuple(p for f in self.stage for p in f.consumes)

    def _output_names(self, base_columns: Iterable[str]) -> list[str]:
        consumed = self._consumed()
        names = [c for c in base_columns if not (consumed and c.startswith(consumed))]
        names += [n for f in self.stage for n in f.output_names]
        repeated = sorted(n for n, k in Counter(names).items() if k > 1)
        if repeated:
            raise FeatureConflictError(f"{self._name}: repeated columns {repeated}")
        return names

    def clone(self) -> StagedFeatureSet[BattleT]:
        """Independent, unfitted copy."""
        stage = [copy.deepcopy(f) for f in self.stage]
        for f in stage:
            f.reset()
        return StagedFeatureSet[BattleT](self._name, self.base.clone(), stage)

    def fit(
        self,
        battles: Iterable[BattleT],
        labels: Sequence[int] | None = None,
        unlabeled: Iterable[BattleT] = (),
    ) -> StagedFeatureSet[BattleT]:
        """Fit the first stage, then every table feature on its table; return ``self``."""
        battles, extra = list(battles), list(unlabeled)
        self.base.fit(battles, labels, extra)
        self._fit_stage(self.base.to_frame(battles), extra, labels)
        return self

    def _fit_stage(
        self, table: pd.DataFrame, extra: list[BattleT], labels: Sequence[int] | None
    ) -> None:
        """Label-aware features on the labelled rows, label-free ones also on ``extra``'s."""
        free = [f for f in self.stage if f.requires_fit and not f.uses_labels]
        rows = table
        if extra and free:
            rows = pd.concat([table, self.base.to_frame(extra)], ignore_index=True)
        for f in self.stage:
            if not f.requires_fit:
                continue
            if f.uses_labels:
                if labels is None:
                    raise ValueError(f"feature {f.feature_id!r} needs the training labels")
                f.fit(table, labels)
            else:
                f.fit(rows)

    def fit_transform(
        self,
        battles: Iterable[BattleT],
        labels: Sequence[int] | None = None,
        unlabeled: Iterable[BattleT] = (),
        groups: Sequence[Any] | None = None,
        inner_folds: int = DEFAULT_INNER_FOLDS,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Fit on the training data and return its table, label-aware columns out of fold.

        The first stage as ``FeatureSet.fit_transform``; each label-aware table feature then
        gives every inner fold's rows from a copy fitted on the other folds' rows (the folds of
        ``FeatureSet``: ``groups``, ``inner_folds``, ``seed``). Other columns come from the fit
        on all rows, which is the state used later on new battles.
        """
        battles, extra = list(battles), list(unlabeled)
        table = self.base.fit_transform(battles, labels, extra, groups, inner_folds, seed)
        self._fit_stage(table, extra, labels)
        outputs = {f.feature_id: f.transform(table) for f in self.stage}
        aware = [f for f in self.stage if f.uses_labels and f.inner_oof]
        if aware and labels is not None:
            folds = np.array(_inner_fold_ids(len(battles), groups, inner_folds, seed))
            y = np.asarray(labels)
            for fold in sorted(set(folds.tolist())):
                inside, outside = np.flatnonzero(folds == fold), np.flatnonzero(folds != fold)
                for f in aware:
                    twin = copy.deepcopy(f)
                    twin.reset()
                    twin.fit(table.iloc[outside].reset_index(drop=True), y[outside].tolist())
                    rows = table.iloc[inside].reset_index(drop=True)
                    outputs[f.feature_id][inside] = twin.transform(rows)
        return self._assemble(table, outputs)

    def to_frame(
        self,
        battles: Iterable[BattleT],
        index: Callable[[BattleT], Any] | None = None,
    ) -> pd.DataFrame:
        """One row per battle: the first stage's table, then the fitted table features."""
        table = self.base.to_frame(battles, index)
        return self._assemble(table, {f.feature_id: f.transform(table) for f in self.stage})

    def _assemble(self, table: pd.DataFrame, outputs: dict[str, np.ndarray]) -> pd.DataFrame:
        names = self._output_names(table.columns)
        consumed = self._consumed()
        kept = [c for c in table.columns if not (consumed and c.startswith(consumed))]
        blocks = [table.loc[:, kept]] + [
            pd.DataFrame(outputs[f.feature_id], columns=list(f.output_names), index=table.index)
            for f in self.stage
        ]
        out = pd.concat(blocks, axis=1)
        assert list(out.columns) == names
        return out

    def get_state(self) -> dict[str, Any]:
        """The first stage's state and every fitted table feature's."""
        if self.requires_fit:
            raise FeatureNotFittedError("fit() the feature set before saving it")
        stage = {f.feature_id: f.get_state() for f in self.stage if f.requires_fit}
        return {"base": self.base.get_state(), "stage": stage}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore the state saved by :meth:`get_state`."""
        expected = {f.feature_id for f in self.stage if f.requires_fit}
        if set(state["stage"]) != expected:
            raise ValueError(f"saved state is for {sorted(state['stage'])}, expected {expected}")
        self.base.set_state(state["base"])
        for f in self.stage:
            if f.requires_fit:
                f.set_state(state["stage"][f.feature_id])

    def __repr__(self) -> str:
        return f"StagedFeatureSet({self._name!r}, {len(self.feature_ids)} features)"
