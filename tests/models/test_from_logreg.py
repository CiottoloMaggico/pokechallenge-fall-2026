"""gb_lr: same model as the bare booster without NaN, usable with NaN."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from models.core.from_logreg import logreg_pipeline
from models.gb_lr import SKLEARN_PARAMS
from models.gb_lr import build as build_sklearn


def table() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(400, 4)), columns=list("abcd"))
    return x, (x["a"] + rng.normal(size=400) > 0).astype(int).to_numpy()


def test_imputer_changes_nothing_without_nan() -> None:
    x, y = table()
    bare = GradientBoostingClassifier(init=logreg_pipeline(), **SKLEARN_PARAMS).fit(x, y)
    model = build_sklearn().fit(x, y)
    assert np.array_equal(model.predict_proba(x), bare.predict_proba(x)[:, 1])


def test_nan_is_accepted_at_fit_and_predict() -> None:
    x, y = table()
    x.iloc[::10, 0] = np.nan
    p = build_sklearn().fit(x, y).predict_proba(x)
    assert np.isfinite(p).all()
