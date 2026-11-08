"""``hgb_ko``: the gradient boosting of the knockout detector (P(faint) of a switch).

Its parameters are the detector's since the 50-column version. The coded columns of
``features/transitions`` (``x_status``, ``y_code``) are categorical: the model treats them so
when the table has them, so the same model fits ``ko_raw50`` (no codes) and ``ko_domain30``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from interfaces import Model
from models.core.estimator import EstimatorModel
from models.hgb import search_space
from registry import model

PARAMS: dict[str, Any] = {
    "max_iter": 600,
    "learning_rate": 0.03,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "random_state": 42,
}
CATEGORICAL = ("x_status", "y_code")


class CategoricalBoosting(EstimatorModel):
    """HistGradientBoosting whose ``categorical`` columns, if present, are categorical."""

    def __init__(self, name: str, params: Mapping[str, Any], categorical: Sequence[str]) -> None:
        super().__init__(name, HistGradientBoostingClassifier(**params))
        self._params = dict(params)
        self._categorical = tuple(categorical)

    def clone(self) -> CategoricalBoosting:
        return CategoricalBoosting(self.name, self._params, self._categorical)

    def get_params(self) -> dict[str, Any]:
        return {**super().get_params(), "categorical": list(self._categorical)}

    def _fit(self, x: Any, y: np.ndarray) -> None:
        present = [c for c in self._categorical if c in x.columns]
        params = {**self._params, "categorical_features": present or None}
        self._template = HistGradientBoostingClassifier(**params)
        super()._fit(x, y)


@model("hgb_ko", space=search_space)
def build(name: str = "hgb_ko", **params: Any) -> Model:
    return CategoricalBoosting(name, {**PARAMS, **params}, CATEGORICAL)
