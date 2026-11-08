"""The tuner on a registered model's spec and the files of tuned parameters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models import MODELS
from tuning.params import load_params, save_params
from tuning.tuner import Tuner, save_best


def folds() -> list[tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]]:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    y = (x["a"] + rng.normal(size=200) > 0).astype(int).to_numpy()
    return [
        (x.iloc[:150], y[:150], x.iloc[150:], y[150:]),
        (x.iloc[50:], y[50:], x.iloc[:50], y[:50]),
    ]


def test_a_params_file_round_trips(tmp_path: Path) -> None:
    path = save_params(tmp_path / "t" / "m.json", model="hgb", params={"max_iter": 5}, value=0.1)
    assert load_params(path) == ("hgb", {"max_iter": 5})
    assert json.loads(path.read_text())["value"] == 0.1


def test_a_model_without_search_space_is_refused() -> None:
    with pytest.raises(KeyError, match="ensemble"):
        Tuner(MODELS["ensemble"], folds())


def test_a_short_study_saves_the_best_parameters(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    tuner = Tuner(MODELS["logreg"], folds(), metric="log_loss")
    study = tuner.run(n_trials=3, storage=None, study_name="t", seed=0)
    assert len(study.trials) == 3
    saved = save_best(study, tuner, "set", tmp_path / "set__logreg.json")
    model, params = load_params(saved)
    assert (model, params) == ("logreg", study.best_trial.params)
    assert json.loads(saved.read_text())["trials"] == 3
