"""scikit-learn HistGradientBoostingClassifier (handles NaN natively)."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier

from model import Model
from models.preprocessing import tree_model
from models.space import Trial

PARAMS: dict[str, Any] = {
    "max_iter": 1500,
    "learning_rate": 0.01,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 50,
    "l2_regularization": 2.0,
    "max_features": 0.4,
    "random_state": 0,
}


def build(name: str = "hgb", **params: Any) -> Model:
    return tree_model(name, HistGradientBoostingClassifier(**{**PARAMS, **params}))


def search_space(trial: Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 2000, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 4, 63, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 300, log=True),
        "l2_regularization": trial.suggest_float(
            "l2_regularization", 1e-3, 10.0, log=True
        ),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0),
    }
