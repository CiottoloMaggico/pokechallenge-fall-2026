"""Booster that starts from a logistic regression's log-odds (``gb_lr``, ``catboost_lr`` ...).

On the compact feature sets most of the signal is additive: a logistic
regression captures it and tree boosters trained from scratch spend
their capacity re-learning it (and overfit). Here the booster starts
from the logistic regression's log-odds and only learns a correction:

    log-odds(x) = logreg(x) + trees(x)

The corrections should be small: few shallow trees with a low learning
rate. On compact_v5 (fold seeds 42 and 2027) ``gb_lr`` with 300 depth-2
trees at learning rate 0.01 lowered the log loss of the logistic
regression from 0.5420 to 0.5404 and from 0.5437 to 0.5410.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from interfaces import Model
from models.core.estimator import EstimatorModel
from models.core.preprocessing import require

LOGREG_C = 1.0
EPS = 1e-6


def logreg_pipeline(c: float = LOGREG_C) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=c, max_iter=5000),
    )


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class BoostFromLogreg(ClassifierMixin, BaseEstimator):
    """Booster whose starting point is a logistic regression's log-odds.

    ``library``: ``"sklearn"``, ``"catboost"``, ``"lightgbm"`` or
    ``"xgboost"``; ``params`` go to that library's classifier.
    """

    def __init__(
        self, library: str = "sklearn", c: float = LOGREG_C, params: dict | None = None
    ) -> None:
        self.library = library
        self.c = c
        self.params = params

    def fit(self, x: Any, y: Any) -> BoostFromLogreg:
        y = np.asarray(y)
        self.classes_ = np.array([0, 1])
        params = dict(self.params or {})
        if self.library == "sklearn":
            booster = GradientBoostingClassifier(init=logreg_pipeline(self.c), **params)
            # this booster rejects NaN; the median is what its logistic regression imputes too
            self.booster_ = make_pipeline(SimpleImputer(strategy="median"), booster).fit(x, y)
            return self
        self.logreg_ = logreg_pipeline(self.c).fit(x, y)
        margin = _logit(self.logreg_.predict_proba(x)[:, 1])
        if self.library == "catboost":
            cb = require("catboost")
            self.booster_ = cb.CatBoostClassifier(**params)
            self.booster_.fit(cb.Pool(x, y, baseline=margin))
        elif self.library == "lightgbm":
            self.booster_ = require("lightgbm").LGBMClassifier(**params)
            self.booster_.fit(x, y, init_score=margin)
        elif self.library == "xgboost":
            self.booster_ = require("xgboost").XGBClassifier(**params)
            self.booster_.fit(x, y, base_margin=margin)
        else:
            raise ValueError(f"unknown library {self.library!r}")
        return self

    def predict_proba(self, x: Any) -> np.ndarray:
        if self.library == "sklearn":
            return self.booster_.predict_proba(x)
        margin = _logit(self.logreg_.predict_proba(x)[:, 1])
        if self.library == "catboost":
            raw = self.booster_.predict(x, prediction_type="RawFormulaVal")
        elif self.library == "lightgbm":
            raw = self.booster_.predict(x, raw_score=True)
        else:
            raw = self.booster_.predict(x, output_margin=True)
        p = 1.0 / (1.0 + np.exp(-(np.asarray(raw, dtype=float) + margin)))
        return np.column_stack([1 - p, p])


def boost_from_logreg(library: str, defaults: dict[str, Any], name: str, **params: Any) -> Model:
    """Model ``name``: ``defaults`` updated by ``params`` go to the booster, ``c`` to the logreg."""
    c = params.pop("c", LOGREG_C)
    estimator = BoostFromLogreg(library, c, {**defaults, **params})
    return EstimatorModel(name, clone(estimator))
