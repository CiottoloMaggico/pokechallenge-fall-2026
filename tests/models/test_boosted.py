"""The generic boosted members and their logit blend: any table, fit / predict / clone /
save-load, no column name in the parameters, the frozen twins' parameters and weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

import models
from interfaces import Model
from models.boosted import WEIGHTS
from models.core.estimator import EstimatorModel
from models.core.logit_blend import LogitBlend

MEMBERS = ["logreg_l2", "boost_xgboost", "boost_lightgbm", "boost_catboost"]
SMALL = {"boost_xgboost": {"n_estimators": 20}, "boost_lightgbm": {"n_estimators": 20}}
SMALL["boost_catboost"] = {"iterations": 20}


def random_table(n: int = 120) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(n, 3)), columns=["anything", "else", "here"])
    x.iloc[3, 1] = np.nan
    y = (x["anything"] + rng.normal(size=n) > 0).astype(int).to_numpy()
    return x, y


@pytest.mark.parametrize("name", [*MEMBERS, "boost_blend"])
def test_a_model_fits_any_table_and_survives_clone_and_save(name: str, tmp_path: Path) -> None:
    x, y = random_table()
    model = models.build(name, **SMALL.get(name, {}))
    if name == "boost_blend":
        model = LogitBlend(name, [models.build(m, **SMALL.get(m, {})) for m in MEMBERS],
                           [WEIGHTS[m] for m in MEMBERS])  # fmt: skip
    fitted = model.clone().fit(x, y)
    p = fitted.predict_proba(x)
    assert p.shape == (len(x),) and ((0 <= p) & (p <= 1)).all()
    fitted.save(tmp_path / "m.pkl")
    assert np.array_equal(Model.load(tmp_path / "m.pkl").predict_proba(x), p)
    assert not model.clone().is_fitted


def names_in(value: object) -> bool:
    """True if a parameter value holds a string that looks like a column or prefix."""
    if isinstance(value, str):
        return value.endswith("_") or value.startswith(("f_", "k_", "bs_", "bh_", "bm_", "spl_"))
    if isinstance(value, (list, tuple, set)):
        return any(names_in(v) for v in value)
    return False


@pytest.mark.parametrize("name", MEMBERS)
def test_no_column_name_in_the_parameters(name: str) -> None:
    model = models.build(name)
    assert isinstance(model, EstimatorModel)
    params = model.estimator.get_params(deep=True)
    assert not [k for k, v in params.items() if names_in(v)]
    if name == "logreg_l2":
        assert [step for step, _ in model.estimator.steps] == ["imp", "sc", "lr"]
    else:
        assert not isinstance(model.estimator, Pipeline)  # the booster alone


def test_the_parameters_and_weights_are_those_of_the_frozen_twins() -> None:
    from models import v11
    from models.boosted import CATBOOST, LIGHTGBM, XGBOOST

    assert (XGBOOST, LIGHTGBM, CATBOOST) == (v11.XGB, v11.LGBM, v11.CAT)
    assert models.build("boost_blend").get_params() == WEIGHTS
    FROZEN = v11.WEIGHTS

    twins = ["v11_lr_spline", "v11_xgb", "v11_lgbm", "v11_cat"]
    assert list(WEIGHTS.values()) == [FROZEN[t] for t in twins]
