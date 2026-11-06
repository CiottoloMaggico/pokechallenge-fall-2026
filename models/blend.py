"""Blend of other models on the logit scale, with weights learned in CV.

Unlike :class:`models.core.ensemble.EnsembleModel` (a plain average of probabilities),
:class:`LogitBlend` learns how much to trust each member:

1. inside the training data, ``inner_folds`` stratified folds give every
   member out-of-fold probabilities;
2. non-negative weights summing to one are fitted on those, minimising
   the log-loss of ``sigmoid(sum_i w_i * logit(p_i))``;
3. every member is refitted on all the training data.

The weights never see the rows they are scored on: in the gym the blend
is fitted on the training part of each fold like any other model.
Fitting costs ``inner_folds + 1`` fits of each member.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, softmax
from sklearn.model_selection import StratifiedKFold

from interfaces import Model

ArrayLike = Any

INNER_FOLDS = 5
SEED = 0
CLIP = 1e-6


class LogitBlend(Model):
    """Logit-scale weighted average of ``members`` (see the module docstring)."""

    def __init__(
        self,
        name: str,
        members: Sequence[Model],
        inner_folds: int = INNER_FOLDS,
        seed: int = SEED,
    ) -> None:
        super().__init__(name)
        if len(members) < 2:
            raise ValueError(f"{name}: a blend needs at least two models")
        if inner_folds < 2:
            raise ValueError(f"{name}: inner_folds must be at least 2")
        self._members = [member.clone() for member in members]
        self._inner_folds = inner_folds
        self._seed = seed
        self._weights: np.ndarray | None = None

    @property
    def members(self) -> list[Model]:
        return list(self._members)

    @property
    def weights(self) -> list[float] | None:
        """Learned weights, in member order (None before ``fit``)."""
        return None if self._weights is None else self._weights.tolist()

    def clone(self) -> LogitBlend:
        return LogitBlend(self.name, self._members, self._inner_folds, self._seed)

    def get_params(self) -> dict[str, Any]:
        return {
            "members": [member.name for member in self._members],
            "inner_folds": self._inner_folds,
            "weights": self.weights,
        }

    def _fit(self, x: ArrayLike, y: np.ndarray) -> None:
        self._weights = fit_weights(self._out_of_fold(x, y), y)
        for member in self._members:
            member.fit(x, y)

    def _predict_proba(self, x: ArrayLike) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError(f"{self.name}: no weights, call fit() first")
        proba = np.vstack([member.predict_proba(x) for member in self._members])
        return blend(proba, self._weights)

    def _out_of_fold(self, x: ArrayLike, y: np.ndarray) -> np.ndarray:
        """(members, rows) probabilities, each row predicted by models that
        did not see it."""
        oof = np.zeros((len(self._members), len(y)))
        folds = StratifiedKFold(
            self._inner_folds, shuffle=True, random_state=self._seed
        )
        for train, valid in folds.split(np.zeros(len(y)), y):
            x_train, x_valid = _rows(x, train), _rows(x, valid)
            for i, member in enumerate(self._members):
                fitted = member.clone().fit(x_train, y[train])
                oof[i, valid] = fitted.predict_proba(x_valid)
        return oof


def fit_weights(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Weights (>= 0, summing to one) minimising the blend's log-loss.

    ``proba`` has one row per member; the weights are a softmax of free
    parameters, so the constraint holds without a constrained solver.
    """
    logits = logit(np.clip(proba, CLIP, 1 - CLIP))

    def loss(z: np.ndarray) -> float:
        p = np.clip(expit(softmax(z) @ logits), CLIP, 1 - CLIP)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    start = np.zeros(len(proba))
    return softmax(minimize(loss, start, method="L-BFGS-B").x)


def blend(proba: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``sigmoid(sum_i w_i * logit(p_i))`` for every column of ``proba``."""
    return expit(weights @ logit(np.clip(proba, CLIP, 1 - CLIP)))


def _rows(x: ArrayLike, index: np.ndarray) -> ArrayLike:
    return x.iloc[index] if hasattr(x, "iloc") else np.asarray(x)[index]
