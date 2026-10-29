"""LightGBM LGBMClassifier (optional: ``pip install lightgbm``).

The library is imported only when the model is built, so the rest of
the project works without it. The settings mirror ``hgb`` and are not
tuned yet.
"""

from __future__ import annotations

from typing import Any

from model import Model
from models.preprocessing import require, tree_model
from models.space import Trial

LIBRARY = "lightgbm"
PARAMS: dict[str, Any] = {
    "n_estimators": 1500,
    "learning_rate": 0.01,
    "num_leaves": 15,
    "min_child_samples": 50,
    "reg_lambda": 2.0,
    "colsample_bytree": 0.4,
    "subsample": 0.8,
    "subsample_freq": 1,
    "random_state": 0,
    "verbose": -1,
}


def build(name: str = "lightgbm", **params: Any) -> Model:
    module = require(LIBRARY)
    return tree_model(name, module.LGBMClassifier(**{**PARAMS, **params}))


def search_space(trial: Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 63, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }
