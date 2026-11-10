"""A fixed-weight average of the members' logits (the original's ``ensemble.blend``).

``expit(sum_i w_i logit(clip(p_i, 1e-6, 1 - 1e-6)))``, the weights as given (no renormalisation),
unlike ``EnsembleModel``, which averages probabilities over normalised weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from interfaces import Model

CLIP = 1e-6


class LogitBlend(Model):
    """Weighted logit average of fitted-together members."""

    def __init__(self, name: str, members: Sequence[Model], weights: Sequence[float]) -> None:
        super().__init__(name)
        if len(members) != len(weights) or not members:
            raise ValueError(f"{name}: one weight per member is required")
        self._members = list(members)
        self._weights = list(weights)

    def clone(self) -> LogitBlend:
        return LogitBlend(self.name, [m.clone() for m in self._members], self._weights)

    def get_params(self) -> dict[str, Any]:
        return {m.name: w for m, w in zip(self._members, self._weights, strict=True)}

    def _fit(self, x: pd.DataFrame, y: np.ndarray) -> None:
        for member in self._members:
            member.fit(x, y)

    def _predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.stack([m.predict_proba(x) for m in self._members])
        w = np.asarray(self._weights)
        return expit(np.tensordot(w, logit(np.clip(p, CLIP, 1 - CLIP)), axes=1))
