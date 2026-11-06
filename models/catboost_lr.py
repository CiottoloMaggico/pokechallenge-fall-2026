"""catboost_lr: CatBoost with the log-odds as ``baseline``.

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

CATBOOST_PARAMS: dict[str, Any] = {
    "iterations": 300,
    "learning_rate": 0.02,
    "depth": 3,
    "l2_leaf_reg": 10.0,
    "random_seed": 0,
    "verbose": 0,
    "allow_writing_files": False,
}


def search_space(trial: Trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 50, 1500, log=True),
        "depth": trial.suggest_int("depth", 1, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
        "c": trial.suggest_float("c", 0.01, 10.0, log=True),
    }


@model("catboost_lr", space=search_space, requires=("catboost",))
def build(name: str = "catboost_lr", **params: Any) -> Model:
    require("catboost")
    return boost_from_logreg("catboost", CATBOOST_PARAMS, name, **params)
