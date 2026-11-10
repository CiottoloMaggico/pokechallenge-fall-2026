"""The label-aware steps of v11's members as table features of ``v11_fs``.

The faithful v11 members learn two things from the labels inside the model: the logit of the bag
columns (``stack0``) and the team effect (``team``). Here they are ``TableFeature``s of the
second stage of ``v11_fs`` (``staged_feature_set``), reading v11's columns from the first stage's
table: their training rows are out of fold (the grouped inner folds of the set), any other
battle gets the state fitted on all the training rows.

* ``V11BagLogit`` (``stack_logit_0``): the logistic regression (C=0.01) of the bag columns, which
  it consumes.
* ``V11TeamEffect`` (``stack_logit_full``, ``tm*``): ``team_effect.TeamScores`` (the members'
  parameters) on the team keys, which it consumes, against one base logit, output before the
  team columns: the logistic regression (C=0.003) of every non-key column (bags included, the
  boosters' ``stack_full``), cross-fitted on the feature's training rows (``StratifiedKFold(5,
  shuffle, 42)``, as the original), so a training row's residual never uses its own label.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features.core.v11_fs.team_effect import TeamScores
from staged_feature_set import TableFeature

SEED = 42
BAG_C = 0.01
FULL_C = 0.003
BAGS = ("bs_", "bh_", "bm_")
KEYS = "k_"
BASE = "stack_logit_full"
# the team effect of the frozen members (TEAM of models/v11.py)
TEAM: dict[str, Any] = {"a": 1.0, "p2": True, "link": True, "pair": True, "soft": True, "raw": True}
TEAM_COLUMNS = ("tm", "tm_n1", "tm_n2", "tm_pair", "tm_npair", "tm_soft", "tm_nsoft")
TEAM_COLUMNS += ("tm_raw1", "tm_raw2")


def logit_pipeline(C: float) -> Pipeline:
    """Median imputation, standardisation, L2 logistic regression: the regression of the
    frozen ``StackedLogit`` (``models.core.v11_stacking``), built here (no import of models)."""
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=C, max_iter=5000)),
        ]
    )


class V11BagLogit(TableFeature):
    """``stack_logit_0``: the logit of the regression of v11's bag columns."""

    requires_fit = True
    uses_labels = True
    consumes = BAGS

    def __init__(self) -> None:
        super().__init__()
        self.model: Pipeline | None = None

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("stack_logit_0",)

    def reads(self, columns: Sequence[str]) -> list[str]:
        return [c for c in columns if c.startswith(BAGS)]

    def _fit(self, x: pd.DataFrame, y: np.ndarray | None) -> None:
        self.model = logit_pipeline(BAG_C).fit(x, y)

    def _transform(self, x: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        return self.model.decision_function(x)[:, None]


def key_frame(x: pd.DataFrame, base: np.ndarray) -> pd.DataFrame:
    """The ``k_`` columns and the base logit, as ``TeamScores`` reads them."""
    keys = [c for c in x.columns if c.startswith(KEYS)]
    return x.loc[:, keys].reset_index(drop=True).assign(**{BASE: base})


class V11TeamEffect(TableFeature):
    """``stack_logit_full``, the logit of v11's non-key columns, and ``tm*``, the team effect
    against it."""

    requires_fit = True
    uses_labels = True
    consumes = (KEYS,)

    def __init__(self) -> None:
        super().__init__()
        self.scores = TeamScores(BASE, **TEAM)
        self.base: Pipeline | None = None
        self.state: tuple[Any, ...] | None = None

    @property
    def output_names(self) -> tuple[str, ...]:
        return (BASE, *TEAM_COLUMNS)

    def reads(self, columns: Sequence[str]) -> list[str]:
        return list(columns)

    @staticmethod
    def _non_keys(x: pd.DataFrame) -> pd.DataFrame:
        return x.loc[:, [c for c in x.columns if not c.startswith(KEYS)]]

    def _fit(self, x: pd.DataFrame, y: np.ndarray | None) -> None:
        assert y is not None
        inputs = self._non_keys(x)
        base = np.zeros(len(x))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(inputs, y):
            m = logit_pipeline(FULL_C).fit(inputs.iloc[tr], y[tr])
            base[va] = m.decision_function(inputs.iloc[va])
        self.base = logit_pipeline(FULL_C).fit(inputs, y)
        self.state = self.scores.fit(key_frame(x, base), y)

    def _transform(self, x: pd.DataFrame) -> np.ndarray:
        assert self.base is not None and self.state is not None
        base = self.base.decision_function(self._non_keys(x))
        out = self.scores.encode(key_frame(x, base), self.state)
        return np.column_stack([base, *(out[name] for name in TEAM_COLUMNS)])
