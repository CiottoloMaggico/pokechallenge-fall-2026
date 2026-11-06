"""Weighted average of the probabilities of other models."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from interfaces import Model


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

    def _fit(self, x: Any, y: np.ndarray) -> None:
        for member in self._members:
            member.fit(x, y)

    def _predict_proba(self, x: Any) -> np.ndarray:
        stacked = np.column_stack([m.predict_proba(x) for m in self._members])
        return stacked @ np.asarray(self._weights)
