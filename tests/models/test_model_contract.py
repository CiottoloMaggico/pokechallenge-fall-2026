"""Every registered model whose libraries are installed fits, predicts, clones and saves alike."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import models
from interfaces import Model
from tuning import load_params

NAMES = models.available()


def table() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y = (x["a"] + rng.normal(size=200) > 0).astype(int).to_numpy()
    return x, y


# a handful of trees: the contract does not depend on the size of the model
SMALL: dict[str, dict[str, int]] = {
    "hgb": {"max_iter": 20},
    "hgb_ko": {"max_iter": 20},
    "lightgbm": {"n_estimators": 20},
    "xgboost": {"n_estimators": 20},
    "catboost": {"iterations": 20},
    "gb_lr": {"n_estimators": 20},
    "lightgbm_lr": {"n_estimators": 20},
    "xgboost_lr": {"n_estimators": 20},
    "catboost_lr": {"iterations": 20},
}


def fast(name: str) -> Model:
    """The model with few trees; an ensemble is rebuilt from its small members."""
    if name in models.ENSEMBLES:
        return models.build_tuned(name, SMALL)
    return models.build(name, **SMALL.get(name, {}))


def test_there_are_models_to_check() -> None:
    assert {"logreg", "hgb", "gb_lr", "ensemble", "ensemble_from_lr"} <= set(NAMES)
    assert set(NAMES) <= set(models.MODELS)


@pytest.mark.parametrize("name", [n for n in NAMES if n != "blend_v11"])
def test_fit_predict_clone_params_and_save_load(name: str, tmp_path: Path) -> None:
    x, y = table()
    model = fast(name)
    assert model.name.startswith(name)
    fitted = model.clone().fit(x, y)
    proba = fitted.predict_proba(x)
    assert proba.shape == (len(x),)
    assert ((0 <= proba) & (proba <= 1)).all()
    assert set(fitted.predict(x)) <= {0, 1}
    assert not model.clone().is_fitted
    json.dumps(model.get_params(), default=str)

    path = tmp_path / "model.pkl"
    fitted.save(path)
    assert np.array_equal(Model.load(path).predict_proba(x), proba)


def test_blend_v11_blends_with_the_probability_column() -> None:
    x, y = table()
    x["v11_proba"] = np.nan
    blend = models.build("blend_v11", w=0.5, base="logreg").fit(x, y)
    x["v11_proba"] = 0.5
    assert np.isfinite(blend.predict_proba(x)).all()
    assert json.dumps(blend.get_params())


def test_params_files_of_the_tuner_still_build_tuned_models(tmp_path: Path) -> None:
    path = tmp_path / "gb_lr.json"
    path.write_text(json.dumps({"model": "gb_lr", "params": {"learning_rate": 0.07}}))
    tuned = dict([load_params(path)])
    model = models.build_tuned("ensemble_from_lr", tuned)
    assert model.name == "ensemble_from_lr-tuned"
    assert models.build_tuned("gb_lr", tuned).name == "gb_lr-tuned"
