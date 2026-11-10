"""v11's spline logistic regression and cross-fitted stacked logit.

A copy of ``PrefixScale``, ``spline_lr`` and ``StackedLogit`` of the original's ``models.py``,
same constants and operation order, made generic by contract: ``StackedLogit`` picks its block
at fit time (columns with given prefixes, or all but some prefixes, in the table's order) and
passes the table through when the block is empty; ``spline_lr`` keeps only the keys it is given
(none: no spline block) and ``SplineLogit`` gives it the keys present in the table it fits.
``StackedLogit`` consolidates the frame after dropping its block (same values, no pandas
fragmentation). On the ``v11`` table every step is active and the values are the original's.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

SEED = 42
SPLINE_KEYS = [
    "f_d_fainted",
    "f_d_hp_sum",
    "f_p1_fainted",
    "f_p2_fainted",
    "f_p1_hp_sum",
    "f_p2_hp_sum",
    "f_d_strength",
    "t_hd_mean",
    "t_fd_mean",
    "a_ttk_adv",
    "m_win_mean",
    "f_d_alive",
    "a_a1_hp",
    "a_a2_hp",
]
BAG_PREFIXES = ("bs_", "bh_", "bm_", "bt_", "bc_")


class PrefixScale(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Multiplies standardized columns with the given prefixes, i.e. a stronger L2 penalty on
    them (identity when ``scale`` is 1)."""

    def __init__(self, prefixes: tuple[str, ...] = BAG_PREFIXES, scale: float = 1.0) -> None:
        self.prefixes, self.scale = prefixes, scale

    def fit(self, X: pd.DataFrame, y: Any = None) -> PrefixScale:
        """Record the number of input columns."""
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale the prefixed columns."""
        if self.scale == 1.0:
            return X
        X = X.copy()
        cols = [c for c in X.columns if c.startswith(self.prefixes)]
        X[cols] = X[cols] * self.scale
        return X


def spline_lr(
    C: float = 0.0007,
    n_knots: int = 4,
    bag_scale: float = 1.0,
    keys: Sequence[str] | None = None,
) -> Pipeline:
    """L2 logistic regression with cubic splines of the ``keys`` columns (median-imputed), the
    other columns passed through; everything median-imputed and standardized."""
    keys = SPLINE_KEYS if keys is None else list(keys)
    spl = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("s", SplineTransformer(n_knots=n_knots, degree=3)),
        ]
    )
    splines = ColumnTransformer(
        [("spl", spl, keys)], remainder="passthrough", verbose_feature_names_out=False
    )
    pipe = Pipeline(
        [
            *([("ct", splines)] if keys else []),
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("scale", PrefixScale(scale=bag_scale)),
            ("lr", LogisticRegression(C=C, max_iter=5000)),
        ]
    )
    return pipe.set_output(transform="pandas")


class SplineLogit(ClassifierMixin, BaseEstimator):
    """``spline_lr`` on the ``keys`` present in the table it is fitted on."""

    def __init__(
        self,
        C: float = 0.0007,
        n_knots: int = 4,
        bag_scale: float = 1.0,
        keys: Sequence[str] | None = None,
    ) -> None:
        self.C, self.n_knots, self.bag_scale, self.keys = C, n_knots, bag_scale, keys

    def fit(self, X: pd.DataFrame, y: Any) -> SplineLogit:
        """Fit ``spline_lr`` with the keys found in ``X``."""
        wanted = SPLINE_KEYS if self.keys is None else self.keys
        keys = [k for k in wanted if k in X.columns]
        self.pipeline_ = spline_lr(self.C, self.n_knots, self.bag_scale, keys).fit(X, y)
        self.classes_ = self.pipeline_.classes_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(X)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.decision_function(X)


class StackedLogit(BaseEstimator, TransformerMixin):
    """Replaces a block of columns with the logit of an L2 logistic regression fitted on them.

    The block is resolved at fit time: the columns starting with ``prefixes`` (every column when
    None) and not with ``exclude``, in the table's order; an empty block leaves the table as it
    is. ``fit_transform`` returns cross-fitted values (stratified 5-fold, grouped by the index
    when it has duplicates), ``transform`` uses the model fitted on all rows. With ``keep`` the
    block stays; with ``spline_keys`` the regression is ``spline_lr`` on the keys in the block.
    """

    def __init__(
        self,
        prefixes: tuple[str, ...] | None = None,
        exclude: tuple[str, ...] = (),
        C: float = 0.05,
        n_splits: int = 5,
        seed: int = SEED,
        name: str = "stack_logit",
        keep: bool = False,
        spline_keys: Sequence[str] | None = None,
    ) -> None:
        self.prefixes, self.exclude = prefixes, exclude
        self.C, self.n_splits, self.seed = C, n_splits, seed
        self.name, self.keep, self.spline_keys = name, keep, spline_keys

    def _resolve(self, X: pd.DataFrame) -> list[str]:
        self.columns_ = [
            c
            for c in X.columns
            if (self.prefixes is None or c.startswith(self.prefixes))
            and not c.startswith(self.exclude)
        ]
        return self.columns_

    def _model(self) -> Pipeline:
        if self.spline_keys:
            return spline_lr(C=self.C, keys=[k for k in self.spline_keys if k in self.columns_])
        return Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(C=self.C, max_iter=5000)),
            ]
        )

    def fit(self, X: pd.DataFrame, y: Any) -> StackedLogit:
        """Fit the regression on all rows (nothing when the block is empty)."""
        if self._resolve(X):
            self.model_ = self._model().fit(X[self.columns_], y)
        return self

    def fit_transform(self, X: pd.DataFrame, y: Any = None, **kw: Any) -> pd.DataFrame:
        """Attach the out-of-fold logit, then fit the regression on all rows."""
        if not self._resolve(X):
            return X
        Z = X[self.columns_]
        out = np.zeros(len(X))
        if X.index.is_unique:
            splits = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed).split(
                Z, y
            )
        else:
            splits = StratifiedGroupKFold(
                self.n_splits, shuffle=True, random_state=self.seed
            ).split(Z, y, X.index)
        for tr, va in splits:
            m = self._model().fit(Z.iloc[tr], y[tr])
            out[va] = m.decision_function(Z.iloc[va])
        self.model_ = self._model().fit(Z, y)
        return self._attach(X, out)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Attach the logit of the regression fitted on all rows."""
        if not self.columns_:
            return X
        return self._attach(X, self.model_.decision_function(X[self.columns_]))

    def _attach(self, X: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
        # dropping scattered columns splits the frame into hundreds of blocks: copy() joins them
        X = X if self.keep else X.drop(columns=self.columns_).copy()
        return X.assign(**{self.name: score})
