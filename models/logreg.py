"""L1 logistic regression on imputed, standardised features."""

from __future__ import annotations

from typing import Any

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from interfaces import Model
from models.core.estimator import EstimatorModel
from models.space import Trial
from registry import model

PARAMS: dict[str, Any] = {
    "C": 0.05,
    "l1_ratio": 1.0,
    "solver": "liblinear",
    "max_iter": 5000,
    "random_state": 0,  # liblinear shuffles its coordinates
}


def search_space(trial: Trial) -> dict[str, Any]:
    """Regularisation strength and type (L1 or L2)."""
    return {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "l1_ratio": trial.suggest_categorical("l1_ratio", [0.0, 1.0]),
    }


@model("logreg", space=search_space)
def build(name: str = "logreg", **params: Any) -> Model:
    """The imputer (median) also drops columns that are entirely NaN."""
    estimator = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(**{**PARAMS, **params}),
    )
    return EstimatorModel(name, estimator)
