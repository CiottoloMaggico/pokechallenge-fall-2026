"""Common interface for binary classifiers used in experiments.

Every learner is wrapped in a :class:`Model`, so the training loop (the
"gym") can fit, evaluate and compare models without knowing which
library is behind each one.

* :class:`Model` is the abstract interface the gym depends on.
* :class:`EstimatorModel` adapts any estimator with a scikit-learn style
  API (scikit-learn, XGBoost, LightGBM, CatBoost, ...).
* :class:`EnsembleModel` averages the probabilities of other models.

Example::

    models = [
        EstimatorModel(
            "logreg",
            make_pipeline(StandardScaler(), LogisticRegression(C=0.05)),
        ),
        EstimatorModel("hgb", HistGradientBoostingClassifier()),
    ]
    models.append(EnsembleModel("logreg+hgb", models))

    for model in models:
        fitted = model.clone().fit(x_train, y_train)
        accuracy = (fitted.predict(x_valid) == y_valid).mean()
"""

from __future__ import annotations

import copy
import math
import pickle
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

ArrayLike = Any


class NotFittedError(RuntimeError):
    """``predict`` or ``predict_proba`` called before ``fit``."""


class Model(ABC):
    """A binary classifier: positive class = Player 1 wins (label 1)."""

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("a model needs a non-empty name")
        self._name = name
        self._fitted = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, x: ArrayLike, y: ArrayLike) -> Model:
        """Train on ``x`` and binary labels ``y``; return ``self``."""
        labels = np.asarray(y)
        _check_binary(labels)
        self._fit(x, labels)
        self._fitted = True
        return self

    def predict_proba(self, x: ArrayLike) -> np.ndarray:
        """Return P(label = 1) for each row, as a 1-D array."""
        self._require_fitted()
        proba = np.asarray(self._predict_proba(x), dtype=float).ravel()
        if proba.size and (proba.min() < 0 or proba.max() > 1):
            raise ValueError(f"{self.name}: probabilities outside [0, 1]")
        return proba

    def predict(self, x: ArrayLike, threshold: float = 0.5) -> np.ndarray:
        """Return hard 0/1 predictions (1 when P(label = 1) > threshold)."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return (self.predict_proba(x) > threshold).astype(int)

    @abstractmethod
    def clone(self) -> Model:
        """Return an unfitted copy with the same configuration."""

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return the configuration, for logging and reports."""

    def save(self, path: str | Path) -> None:
        """Serialize the model (fitted or not) with pickle."""
        with Path(path).open("wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> Model:
        """Load a model saved with :meth:`save` (only trusted files)."""
        with Path(path).open("rb") as fh:
            model = pickle.load(fh)
        if not isinstance(model, Model):
            raise TypeError(f"{path} does not contain a Model")
        return model

    def __repr__(self) -> str:
        state = "fitted" if self._fitted else "unfitted"
        return f"{type(self).__name__}({self.name!r}, {state})"

    @abstractmethod
    def _fit(self, x: ArrayLike, y: np.ndarray) -> None:
        """Train the underlying learner."""

    @abstractmethod
    def _predict_proba(self, x: ArrayLike) -> ArrayLike:
        """Return P(label = 1) for each row."""

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError(f"{self.name}: call fit() first")


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
        if not (
            hasattr(estimator, "predict_proba")
            or hasattr(estimator, "decision_function")
        ):
            raise TypeError(
                f"{name}: the estimator needs predict_proba() or decision_function()"
            )
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

    def _fit(self, x: ArrayLike, y: np.ndarray) -> None:
        x = self._select(x)
        self._columns = _column_names(x)
        self._estimator = _clone_estimator(self._template)
        self._estimator.fit(x, y)

    def _predict_proba(self, x: ArrayLike) -> np.ndarray:
        x = self._select(x)
        self._check_columns(x)
        if hasattr(self._estimator, "predict_proba"):
            proba = np.asarray(self._estimator.predict_proba(x))
            return proba[:, self._positive_index()]
        scores = np.asarray(self._estimator.decision_function(x))
        return 1.0 / (1.0 + np.exp(-scores))

    def _select(self, x: ArrayLike) -> ArrayLike:
        if self._features is None:
            return x
        if not hasattr(x, "loc"):
            raise TypeError(f"{self.name}: a feature subset needs a pandas DataFrame")
        missing = [c for c in self._features if c not in x.columns]
        if missing:
            raise KeyError(f"{self.name}: missing features {missing[:5]}")
        return x.loc[:, self._features]

    def _check_columns(self, x: ArrayLike) -> None:
        columns = _column_names(x)
        known = self._columns is not None and columns is not None
        if known and columns != self._columns:
            raise ValueError(f"{self.name}: columns differ from the ones seen in fit")

    def _positive_index(self) -> int:
        classes = list(getattr(self._estimator, "classes_", [0, 1]))
        return classes.index(1) if 1 in classes else len(classes) - 1


class EnsembleModel(Model):
    """Weighted average of the probabilities of other models."""

    def __init__(
        self,
        name: str,
        members: Sequence[Model],
        weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__(name)
        if not members:
            raise ValueError(f"{name}: an ensemble needs at least one model")
        if weights is None:
            weights = [1.0] * len(members)
        if len(weights) != len(members):
            raise ValueError(f"{name}: one weight per member is required")
        if any(w < 0 for w in weights) or math.isclose(sum(weights), 0.0):
            raise ValueError(f"{name}: weights must be >= 0, not all zero")
        self._members = [m.clone() for m in members]
        total = float(sum(weights))
        self._weights = [w / total for w in weights]

    @property
    def members(self) -> list[Model]:
        return list(self._members)

    def clone(self) -> EnsembleModel:
        return EnsembleModel(self.name, self._members, self._weights)

    def get_params(self) -> dict[str, Any]:
        return {
            "members": [m.name for m in self._members],
            "weights": list(self._weights),
        }

    def _fit(self, x: ArrayLike, y: np.ndarray) -> None:
        for member in self._members:
            member.fit(x, y)

    def _predict_proba(self, x: ArrayLike) -> np.ndarray:
        stacked = np.column_stack([m.predict_proba(x) for m in self._members])
        return stacked @ np.asarray(self._weights)


def _check_binary(y: np.ndarray) -> None:
    values = set(np.unique(y).tolist())
    if not values <= {0, 1} or len(values) < 2:
        raise ValueError(f"labels must contain both 0 and 1, got {values}")


def _column_names(x: ArrayLike) -> list[str] | None:
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
