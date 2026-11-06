"""xgboost_lr: XGBoost with the log-odds as ``base_margin``.

Gradient boosting that learns only a small correction of a logistic regression
(``models/core/from_logreg.py``): few shallow trees with a low learning rate.
"""

from __future__ import annotations

from typing import Any

from interfaces import Model
from models.core.from_logreg import boost_from_logreg
from models.core.preprocessing import require
from models.space import Trial
from registry import model

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


def search_space(trial: Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1500, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 4),
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.1, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
        "c": trial.suggest_float("c", 0.01, 10.0, log=True),
    }


@model("xgboost_lr", space=search_space, requires=("xgboost",))
def build(name: str = "xgboost_lr", **params: Any) -> Model:
    require("xgboost")
    return boost_from_logreg("xgboost", XGBOOST_PARAMS, name, **params)
