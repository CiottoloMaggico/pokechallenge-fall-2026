"""``lgbm_ko_v11``: the LightGBM of v11's hidden-faint detector (P(faint) of a switch).

The parameters of the original's ``faint.py`` (seed 42, four threads, deterministic, row-wise
histograms); no column is dropped, unlike the repository's tree models.
"""

from __future__ import annotations

from typing import Any

from interfaces import Model
from models.core.estimator import EstimatorModel
from models.core.preprocessing import require
from registry import model

SEED = 42
N_JOBS = 4
PARAMS: dict[str, Any] = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "deterministic": True,
    "force_row_wise": True,
    "random_state": SEED,
    "n_jobs": N_JOBS,
    "verbose": -1,
}


@model("lgbm_ko_v11", requires=("lightgbm",))
def build(name: str = "lgbm_ko_v11", **params: Any) -> Model:
    """The detector's LightGBM; ``params`` override the original's."""
    return EstimatorModel(name, require("lightgbm").LGBMClassifier(**{**PARAMS, **params}))
