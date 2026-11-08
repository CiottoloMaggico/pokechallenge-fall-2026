"""Hyperparameter search with Optuna on fixed, leakage-safe folds.

The feature tables of every fold are built once (``Gym.fold_tables``),
so each trial only fits the model: hundreds of trials cost little more
than the features. Each trial reports its running mean after every fold,
so Optuna can stop bad trials early (median pruning).

With a storage URL (``sqlite:///results/tuning/optuna.db``) a study survives
restarts and several processes can work on it at once: start the same
``tune`` command in more terminals with the same study name.

Example::

    folds = Gym.fold_tables(train.labelled(), feature_set, FoldPlan(5, (2027,)))[2027]
    tuner = Tuner(MODELS["hgb"], folds, metric="log_loss")
    study = tuner.run(n_trials=200, storage="sqlite:///results/tuning/optuna.db",
                      study_name="compact__hgb")
    save_best(study, tuner, "compact", Path("results/tuning/compact__hgb.json"))
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from folds import FoldData
from models.core.preprocessing import require
from registry import ModelSpec
from tuning.params import save_params

PROBA_CLIP = 1e-6
METRICS: dict[str, tuple[str, Callable[[np.ndarray, np.ndarray], float]]] = {
    "accuracy": ("maximize", lambda y, p: float(accuracy_score(y, p > 0.5))),
    "log_loss": (
        "minimize",
        lambda y, p: float(log_loss(y, np.clip(p, PROBA_CLIP, 1 - PROBA_CLIP))),
    ),
}


class Tuner:
    """Optuna objective for one registered model on precomputed folds."""

    def __init__(
        self, spec: ModelSpec, folds: Sequence[FoldData], metric: str = "log_loss"
    ) -> None:
        if spec.search_space is None:
            raise KeyError(f"no search space for {spec.name!r}")
        self.space = spec.search_space
        if metric not in METRICS:
            raise KeyError(f"unknown metric {metric!r}; choose from {sorted(METRICS)}")
        if not folds:
            raise ValueError("at least one fold is required")
        self.spec = spec
        self.model_name = spec.name
        self.folds = list(folds)
        self.metric = metric
        self.direction, self._score = METRICS[metric]

    def objective(self, trial: Any) -> float:
        """Mean ``metric`` over the folds for the parameters ``trial`` picks."""
        params = self.space(trial)
        scores, accuracies = [], []
        for step, (x_train, y_train, x_valid, y_valid) in enumerate(self.folds):
            model = self.spec.build(**params)
            proba = model.fit(x_train, y_train).predict_proba(x_valid)
            scores.append(self._score(y_valid, proba))
            accuracies.append(METRICS["accuracy"][1](y_valid, proba))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise require("optuna").TrialPruned()
        trial.set_user_attr("accuracy", float(np.mean(accuracies)))
        trial.set_user_attr("accuracy_std", float(np.std(accuracies)))
        return float(np.mean(scores))

    def run(
        self,
        n_trials: int | None = None,
        timeout: float | None = None,
        storage: str | None = None,
        study_name: str | None = None,
        seed: int = 0,
    ) -> Any:
        """Create (or resume) the study and optimise it."""
        optuna = require("optuna")
        optuna.logging.set_verbosity(optuna.logging.INFO)  # one line per trial
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        )
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)
        return study


def save_best(study: Any, tuner: Tuner, feature_set: str, path: Path) -> Path:
    """Write the best trial as a parameter file for ``main.py --params``."""
    best = study.best_trial
    return save_params(
        path,
        model=tuner.model_name,
        feature_set=feature_set,
        params=best.params,
        metric=tuner.metric,
        value=best.value,
        accuracy=best.user_attrs.get("accuracy"),
        accuracy_std=best.user_attrs.get("accuracy_std"),
        trials=len(study.trials),
        saved=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
