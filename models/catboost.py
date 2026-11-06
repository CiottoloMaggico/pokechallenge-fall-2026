"""CatBoost CatBoostClassifier (optional: ``pip install catboost``).

The library is imported only when the model is built, so the rest of
the project works without it. The settings mirror ``hgb`` and are not
tuned yet.
"""

from __future__ import annotations

from typing import Any

from interfaces import Model
from models.core.preprocessing import require, tree_model
from models.space import Trial
from registry import model

PARAMS: dict[str, Any] = {
    "iterations": 1500,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "rsm": 0.4,
    "random_seed": 0,
    "verbose": 0,
    "allow_writing_files": False,
}


def search_space(trial: Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "iterations": trial.suggest_int("iterations", 100, 2000, log=True),
        "depth": trial.suggest_int("depth", 2, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "rsm": trial.suggest_float("rsm", 0.2, 1.0),
    }


@model("catboost", space=search_space, requires=("catboost",))
def build(name: str = "catboost", **params: Any) -> Model:
    module = require("catboost")
    return tree_model(name, module.CatBoostClassifier(**{**PARAMS, **params}))
