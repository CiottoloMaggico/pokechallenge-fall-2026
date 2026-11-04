"""Gradient boosting that starts from a logistic regression.

On the compact feature sets most of the signal is additive: a logistic
regression captures it and tree boosters trained from scratch spend
their capacity re-learning it (and overfit). Here the booster starts
from the logistic regression's log-odds and only learns a correction:

    log-odds(x) = logreg(x) + trees(x)

* ``gb_lr``       scikit-learn GradientBoostingClassifier with the
                  logistic regression as ``init`` (no extra library)
* ``catboost_lr`` CatBoost with the log-odds as ``baseline``
* ``lightgbm_lr`` LightGBM with the log-odds as ``init_score``
* ``xgboost_lr``  XGBoost with the log-odds as ``base_margin``

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

from model import EstimatorModel, Model
from models.preprocessing import require
from models.space import Trial

LOGREG_C = 1.0
EPS = 1e-6

# small corrections: few shallow trees, low learning rate
SKLEARN_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 2,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "min_samples_leaf": 50,
    "random_state": 0,
}
CATBOOST_PARAMS: dict[str, Any] = {
    "iterations": 300,
    "learning_rate": 0.02,
    "depth": 3,
    "l2_leaf_reg": 10.0,
    "random_seed": 0,
    "verbose": 0,
    "allow_writing_files": False,
}
LIGHTGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.01,
    "num_leaves": 4,
    "min_child_samples": 50,
    "reg_lambda": 5.0,
    "subsample": 0.8,
    "subsample_freq": 1,
    "random_state": 0,
    "verbose": -1,
}
XGBOOST_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.01,
    "max_depth": 2,
    "min_child_weight": 10,
    "reg_lambda": 5.0,
    "subsample": 0.8,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "random_state": 0,
}


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
            init = logreg_pipeline(self.c)
            self.booster_ = GradientBoostingClassifier(init=init, **params).fit(x, y)
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


def _build(library: str, defaults: dict[str, Any], name: str, **params: Any) -> Model:
    c = params.pop("c", LOGREG_C)
    estimator = BoostFromLogreg(library, c, {**defaults, **params})
    return EstimatorModel(name, clone(estimator))


def build_sklearn(name: str = "gb_lr", **params: Any) -> Model:
    return _build("sklearn", SKLEARN_PARAMS, name, **params)


def build_catboost(name: str = "catboost_lr", **params: Any) -> Model:
    require("catboost")
    return _build("catboost", CATBOOST_PARAMS, name, **params)


def build_lightgbm(name: str = "lightgbm_lr", **params: Any) -> Model:
    require("lightgbm")
    return _build("lightgbm", LIGHTGBM_PARAMS, name, **params)


def build_xgboost(name: str = "xgboost_lr", **params: Any) -> Model:
    require("xgboost")
    return _build("xgboost", XGBOOST_PARAMS, name, **params)


# search spaces: only small corrections (shallow trees, low learning rates)
def search_space_sklearn(trial: Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 800, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 4),
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.1, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "c": trial.suggest_float("c", 0.01, 10.0, log=True),
    }


def search_space_catboost(trial: Trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 50, 1500, log=True),
        "depth": trial.suggest_int("depth", 1, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
        "c": trial.suggest_float("c", 0.01, 10.0, log=True),
    }


def search_space_lightgbm(trial: Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1500, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 16, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
        "c": trial.suggest_float("c", 0.01, 10.0, log=True),
    }


def search_space_xgboost(trial: Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1500, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 4),
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.1, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
        "c": trial.suggest_float("c", 0.01, 10.0, log=True),
    }
