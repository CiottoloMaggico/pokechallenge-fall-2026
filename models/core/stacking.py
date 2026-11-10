"""A cross-fitted stacked logit of every column, kept next to them (for the generic models).

The logic of the frozen ``models.core.v11_stacking.StackedLogit`` without column selectors or
splines: the regression (median imputation, standardisation, L2 ``LogisticRegression``) sees
the whole table, whatever its columns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def logit_pipeline(C: float) -> Pipeline:
    """Median imputation, standardisation, L2 logistic regression (``max_iter`` 5000)."""
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=C, max_iter=5000)),
        ]
    )


class StackedLogit(BaseEstimator, TransformerMixin):
    """Adds the logit ``name`` of a logistic regression fitted on every column.

    ``fit_transform`` returns cross-fitted values (stratified ``n_splits``-fold, shuffled with
    ``seed``, grouped by the index when it has duplicates), ``transform`` uses the regression
    fitted on all rows. With ``keep`` the columns stay, otherwise only the logit is left.
    """

    def __init__(
        self,
        C: float = 0.003,
        n_splits: int = 5,
        seed: int = SEED,
        name: str = "stack_logit",
        keep: bool = True,
    ) -> None:
        self.C, self.n_splits, self.seed, self.name, self.keep = C, n_splits, seed, name, keep

    def fit(self, X: pd.DataFrame, y: Any) -> StackedLogit:
        """Fit the regression on all rows."""
        self.model_ = logit_pipeline(self.C).fit(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y: Any = None, **kw: Any) -> pd.DataFrame:
        """Attach the out-of-fold logit, then fit the regression on all rows."""
        y = np.asarray(y)
        out = np.zeros(len(X))
        if X.index.is_unique:
            splits = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed).split(
                X, y
            )
        else:
            splits = StratifiedGroupKFold(
                self.n_splits, shuffle=True, random_state=self.seed
            ).split(X, y, X.index)
        for tr, va in splits:
            m = logit_pipeline(self.C).fit(X.iloc[tr], y[tr])
            out[va] = m.decision_function(X.iloc[va])
        self.model_ = logit_pipeline(self.C).fit(X, y)
        return self._attach(X, out)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Attach the logit of the regression fitted on all rows."""
        return self._attach(X, self.model_.decision_function(X))

    def _attach(self, X: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
        base = X if self.keep else pd.DataFrame(index=X.index)
        return base.assign(**{self.name: score})
