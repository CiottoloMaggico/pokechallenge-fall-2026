"""lightgbm_lr: LightGBM with the log-odds as ``init_score``.

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


def search_space(trial: Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1500, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 16, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
        "c": trial.suggest_float("c", 0.01, 10.0, log=True),
    }


@model("lightgbm_lr", space=search_space, requires=("lightgbm",))
def build(name: str = "lightgbm_lr", **params: Any) -> Model:
    require("lightgbm")
    return boost_from_logreg("lightgbm", LIGHTGBM_PARAMS, name, **params)
