"""XGBoost XGBClassifier (optional: ``pip install xgboost``).

The library is imported only when the model is built, so the rest of
the project works without it. The settings mirror ``hgb`` and are not
tuned yet.
"""

from __future__ import annotations

from typing import Any

from model import Model
from models.preprocessing import require, tree_model
from models.space import Trial

LIBRARY = "xgboost"
PARAMS: dict[str, Any] = {
    "n_estimators": 1500,
    "learning_rate": 0.01,
    "max_depth": 4,
    "min_child_weight": 5,
    "reg_lambda": 2.0,
    "colsample_bytree": 0.4,
    "subsample": 0.8,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "random_state": 0,
}


def build(name: str = "xgboost", **params: Any) -> Model:
    module = require(LIBRARY)
    return tree_model(name, module.XGBClassifier(**{**PARAMS, **params}))


def search_space(trial: Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1.0, 50.0, log=True
        ),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }
