"""Logit blend of a catalog model with a probability column.

``LogitBlend`` fits ``base`` (a catalog model, optionally with tuned
parameters) on every column except ``column`` and predicts

    p = expit(w * logit(p_base) + (1 - w) * logit(column))

``column`` is the probability of another model given by a feature with
``inner_oof = False`` (the v11 bridge's ``v11_proba`` until step 8, a native v11
later), NaN on training battles: it is never used for fitting, only read on
predicted battles; where it is missing (a test battle identical to a training
one) p = p_base. w = 1 is the base alone, w = 0 the column alone. No model is
registered with it since the v11 bridge was removed (REFACTORING.md §10).

``tuned`` lists parameter files of ``main.py tune`` (``results/tuning/*.json``)
for the base model or its ensemble members, applied as ``--params`` would.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from interfaces import Model

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class LogitBlend(Model):
    def __init__(
        self,
        name: str = "blend_v11",
        w: float = 0.5,
        base: str = "ensemble_from_lr",
        column: str = "v11_proba",
        tuned: Sequence[str] = (),
    ) -> None:
        super().__init__(name)
        if not 0.0 <= w <= 1.0:
            raise ValueError("w must be in [0, 1]")
        self.w, self.base, self.column, self.tuned = float(w), base, column, tuple(tuned)
        self._base: Model  # set by fit; Model.predict_proba refuses an unfitted blend

    def _build_base(self) -> Model:
        from registry import build_tuned

        # the files of tuning/params.py, read here because models never import the application
        infos = [json.loads(Path(path).read_text()) for path in self.tuned]
        return build_tuned(self.base, {str(i["model"]): dict(i["params"]) for i in infos})

    def _split(self, x: Any) -> tuple[pd.DataFrame, np.ndarray]:
        if not isinstance(x, pd.DataFrame) or self.column not in x:
            raise KeyError(f"{self.name}: needs a DataFrame with the column {self.column!r}")
        return x.drop(columns=[self.column]), x[self.column].to_numpy(dtype=float)

    def _fit(self, x: Any, y: np.ndarray) -> None:
        features, _ = self._split(x)  # the column is NaN on training rows: unused
        self._base = self._build_base().fit(features, y)

    def _predict_proba(self, x: Any) -> np.ndarray:
        features, other = self._split(x)
        base = self._base.predict_proba(features)
        z = self.w * _logit(base) + (1 - self.w) * _logit(np.where(np.isnan(other), base, other))
        return 1.0 / (1.0 + np.exp(-z))

    def clone(self) -> LogitBlend:
        return LogitBlend(self.name, self.w, self.base, self.column, self.tuned)

    def get_params(self) -> dict[str, Any]:
        return {"w": self.w, "base": self.base, "column": self.column, "tuned": list(self.tuned)}
