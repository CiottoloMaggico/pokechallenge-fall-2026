"""gb_lr: scikit-learn GradientBoostingClassifier with the logistic regression as ``init``.

Gradient boosting that learns only a small correction of a logistic regression
(``models/core/from_logreg.py``): few shallow trees with a low learning rate.
"""

from __future__ import annotations

from typing import Any

from interfaces import Model
from models.core.from_logreg import boost_from_logreg
from models.space import Trial
from registry import model

SKLEARN_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 2,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "min_samples_leaf": 50,
    "random_state": 0,
}


def search_space(trial: Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 800, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 4),
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.1, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "c": trial.suggest_float("c", 0.01, 10.0, log=True),
    }


@model("gb_lr", space=search_space)
def build(name: str = "gb_lr", **params: Any) -> Model:
    return boost_from_logreg("sklearn", SKLEARN_PARAMS, name, **params)
