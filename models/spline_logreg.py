"""Logistic regression on cubic splines of every feature.

Each column is expanded into a B-spline basis (``n_knots`` uniform knots,
cubic, linear beyond the training range), so the regression can bend:
being one Pokémon ahead does not have to be worth the same at every
point of the scale. The features themselves do not change, only how the
model reads them. The basis is standardised and fitted with an L2
penalty, since every feature now has several correlated columns.
"""

from __future__ import annotations

from typing import Any

from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from interfaces import Model
from models.core.estimator import EstimatorModel
from models.space import Trial

N_KNOTS = 4
DEGREE = 3
PARAMS: dict[str, Any] = {"C": 0.005, "max_iter": 5000}


def build(
    name: str = "spline_logreg",
    n_knots: int = N_KNOTS,
    degree: int = DEGREE,
    **params: Any,
) -> Model:
    """Constant columns are dropped first (a spline needs a range)."""
    estimator = make_pipeline(
        SimpleImputer(strategy="median"),
        VarianceThreshold(),
        SplineTransformer(n_knots=n_knots, degree=degree, extrapolation="linear"),
        StandardScaler(),
        LogisticRegression(**{**PARAMS, **params}),
    )
    return EstimatorModel(name, estimator)


def search_space(trial: Trial) -> dict[str, Any]:
    """Regularisation strength and number of knots."""
    return {
        "C": trial.suggest_float("C", 1e-3, 1.0, log=True),
        "n_knots": trial.suggest_int("n_knots", 3, 8),
    }
