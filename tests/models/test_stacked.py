"""The generic stacked members and their logit blend: out-of-fold stacked logit (the frozen v11
step's logic), any table, fit / predict / clone / save-load, no column name in the parameters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

import models
from interfaces import Model
from models.core import v11_stacking
from models.core.estimator import EstimatorModel
from models.core.logit_blend import LogitBlend
from models.core.stacking import StackedLogit
from models.stacked import WEIGHTS

MEMBERS = ["logreg_l2", "stack_xgboost", "stack_lightgbm", "stack_catboost"]
SMALL = {"stack_xgboost": {"n_estimators": 20}, "stack_lightgbm": {"n_estimators": 20}}
SMALL["stack_catboost"] = {"iterations": 20}


def random_table(n: int = 120) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(n, 3)), columns=["anything", "else", "here"])
    x.iloc[3, 1] = np.nan
    y = (x["anything"] + rng.normal(size=n) > 0).astype(int).to_numpy()
    return x, y


def test_the_stacked_logit_is_out_of_fold_and_keeps_the_columns() -> None:
    x, y = random_table()
    stack = StackedLogit(C=0.05, name="s")
    out = stack.fit_transform(x, y)
    assert list(out.columns) == [*x.columns, "s"]
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(x, y):
        fold = StackedLogit(C=0.05).fit(x.iloc[tr], y[tr])
        assert np.array_equal(out["s"].to_numpy()[va], fold.model_.decision_function(x.iloc[va]))
    full = stack.model_.decision_function(x)
    assert np.array_equal(stack.transform(x)["s"].to_numpy(), full)
    assert list(StackedLogit(keep=False).fit_transform(x, y).columns) == ["stack_logit"]


def test_the_stacked_logit_is_the_frozen_step_without_selectors() -> None:
    x, y = random_table()
    frozen = v11_stacking.StackedLogit(C=0.003, name="s", keep=True)
    ours = StackedLogit(C=0.003, name="s")
    assert np.array_equal(ours.fit_transform(x, y), frozen.fit_transform(x, y), equal_nan=True)
    assert np.array_equal(ours.transform(x), frozen.transform(x), equal_nan=True)


@pytest.mark.parametrize("name", [*MEMBERS, "stack_blend"])
def test_a_model_fits_any_table_and_survives_clone_and_save(name: str, tmp_path: Path) -> None:
    x, y = random_table()
    model = models.build(name, **SMALL.get(name, {}))
    if name == "stack_blend":
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
    steps = [step for step, _ in model.estimator.steps]
    assert steps == (["imp", "sc", "lr"] if name == "logreg_l2" else ["stack", "model"])


def test_the_parameters_and_weights_are_those_of_the_frozen_twins() -> None:
    from models import v11
    from models.stacked import CATBOOST, LIGHTGBM, XGBOOST

    assert (XGBOOST, LIGHTGBM, CATBOOST) == (v11.XGB, v11.LGBM, v11.CAT)
    assert models.build("stack_blend").get_params() == WEIGHTS
    FROZEN = v11.WEIGHTS

    twins = ["v11_lr_spline", "v11_xgb", "v11_lgbm", "v11_cat"]
    assert list(WEIGHTS.values()) == [FROZEN[t] for t in twins]
