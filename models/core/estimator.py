"""Adapter that turns any scikit-learn style estimator into a :class:`interfaces.Model`."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np

from interfaces import Model


class EstimatorModel(Model):
    """Adapter for estimators with ``fit`` and ``predict_proba``.

    ``features`` restricts the model to a subset of columns (requires a
    pandas DataFrame), so different models can use different feature
    sets inside the same gym. Column names seen at fit time are checked
    at prediction time, to catch reordered or missing features.

    Estimators without ``predict_proba`` but with ``decision_function``
    (for example ``SVC`` or ``LinearSVC``) are supported through a
    logistic squashing of the decision score; such scores rank rows
    correctly but are not calibrated probabilities.
    """

    def __init__(
        self,
        name: str,
        estimator: Any,
        features: Sequence[str] | None = None,
    ) -> None:
        super().__init__(name)
        if not hasattr(estimator, "fit"):
            raise TypeError(f"{name}: the estimator has no fit() method")
        if not (hasattr(estimator, "predict_proba") or hasattr(estimator, "decision_function")):
            raise TypeError(f"{name}: the estimator needs predict_proba() or decision_function()")
        self._template = estimator
        self._estimator = _clone_estimator(estimator)
        self._features = list(features) if features is not None else None
        self._columns: list[str] | None = None

    @property
    def estimator(self) -> Any:
        """The underlying (possibly fitted) estimator."""
        return self._estimator

    @property
    def features(self) -> list[str] | None:
        """Columns used by the model, or None for all columns."""
        return list(self._features) if self._features else None

    def clone(self) -> EstimatorModel:
        return EstimatorModel(self.name, self._template, self._features)

    def get_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"estimator": type(self._template).__name__}
        if hasattr(self._template, "get_params"):
            params.update(self._template.get_params(deep=False))
        if self._features is not None:
            params["n_features"] = len(self._features)
        return params

    def _fit(self, x: Any, y: np.ndarray) -> None:
        x = self._select(x)
        self._columns = _column_names(x)
        self._estimator = _clone_estimator(self._template)
        self._estimator.fit(x, y)

    def _predict_proba(self, x: Any) -> np.ndarray:
        x = self._select(x)
        self._check_columns(x)
        if hasattr(self._estimator, "predict_proba"):
            proba = np.asarray(self._estimator.predict_proba(x))
            return proba[:, self._positive_index()]
        scores = np.asarray(self._estimator.decision_function(x))
        return 1.0 / (1.0 + np.exp(-scores))

    def _select(self, x: Any) -> Any:
        if self._features is None:
            return x
        if not hasattr(x, "loc"):
            raise TypeError(f"{self.name}: a feature subset needs a pandas DataFrame")
        missing = [c for c in self._features if c not in x.columns]
        if missing:
            raise KeyError(f"{self.name}: missing features {missing[:5]}")
        return x.loc[:, self._features]

    def _check_columns(self, x: Any) -> None:
        columns = _column_names(x)
        known = self._columns is not None and columns is not None
        if known and columns != self._columns:
            raise ValueError(f"{self.name}: columns differ from the ones seen in fit")

    def _positive_index(self) -> int:
        classes = list(getattr(self._estimator, "classes_", [0, 1]))
        return classes.index(1) if 1 in classes else len(classes) - 1


def _column_names(x: Any) -> list[str] | None:
    columns = getattr(x, "columns", None)
    return [str(c) for c in columns] if columns is not None else None


def _clone_estimator(estimator: Any) -> Any:
    """Unfitted copy: scikit-learn ``clone`` when possible, else deepcopy."""
    try:
        from sklearn.base import clone
    except ImportError:
        return copy.deepcopy(estimator)
    try:
        return clone(estimator)
    except TypeError:
        return copy.deepcopy(estimator)
