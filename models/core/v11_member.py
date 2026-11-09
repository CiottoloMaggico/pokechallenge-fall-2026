"""v11's pipelines (one member of ``blend_v11``) and their fixed logit blend.

``V11Member`` builds at fit time, from the columns it receives, the pipeline of the original's
``pipelines.estimator(spec, cols)``: for a tree model, the stacked logit of every non-``k_``
column (kept) and of the bag columns (replacing them); for the spline regression, its own
cross-fitted spline base (dropped after use); then the team effect; then the model of
``models.make``. A step is added only when its columns exist, so the member also fits a table
without them; on the ``v11`` table the pipeline is exactly the original's.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.pipeline import Pipeline

from interfaces import Model
from models.core.preprocessing import require
from models.core.v11_stacking import SEED, StackedLogit, spline_lr
from models.core.v11_team import TeamEffect

N_JOBS = 4
BAG_GROUP_PREFIXES = ("bs_", "bh_", "bm_")
TEAM = {"a": 1.0, "p2": True, "link": True, "pair": True, "soft": True, "raw": True}
FULL_C = 0.003
BAG_C = 0.01
BASE_C = 0.0007
KINDS = ("lr_spline", "xgb", "lgbm", "cat")


def make(kind: str, params: dict[str, Any]) -> Any:
    """The final estimator of a pipeline (the original's ``models.make`` with ``SEED``)."""
    if kind == "lr_spline":
        return spline_lr(**params)
    if kind == "lgbm":
        return require("lightgbm").LGBMClassifier(
            random_state=SEED,
            n_jobs=N_JOBS,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
            **params,
        )
    if kind == "xgb":
        return require("xgboost").XGBClassifier(
            random_state=SEED, n_jobs=N_JOBS, tree_method="hist", eval_metric="logloss", **params
        )
    if kind == "cat":
        return require("catboost").CatBoostClassifier(
            random_seed=SEED, thread_count=N_JOBS, verbose=0, allow_writing_files=False, **params
        )
    raise ValueError(f"unknown v11 model kind {kind!r}")


def pipeline(kind: str, params: dict[str, Any], columns: Sequence[str]) -> Any:
    """The original's ``pipelines.estimator`` of a ``blend_v11`` member on these columns."""
    no_keys = [c for c in columns if not c.startswith("k_")]
    has_team = "k_p1team" in columns
    steps: list[tuple[str, Any]] = []
    if kind == "lr_spline":
        params = {**params, "keys": [k for k in params["keys"] if k in columns]}
        if has_team:
            # the configuration's base_spline is the list of the model's keys
            base = StackedLogit(
                no_keys,
                C=BASE_C,
                seed=SEED,
                name="stack_logit_full",
                keep=True,
                spline_keys=params["keys"],
            )
            steps.append(("team_base", base))
    else:
        full = StackedLogit(no_keys, C=FULL_C, seed=SEED, name="stack_logit_full", keep=True)
        steps.append(("stack_full", full))
        bags = [c for c in columns if c.startswith(BAG_GROUP_PREFIXES)]
        if bags:
            steps.append(("stack0", StackedLogit(bags, C=BAG_C, seed=SEED, name="stack_logit_0")))
    if has_team:
        steps.append(("team", TeamEffect(**TEAM, seed=SEED, drop_base=kind == "lr_spline")))
    est = make(kind, params)
    return Pipeline([*steps, ("model", est)]) if steps else est


class V11Member(Model):
    """One pipeline of ``blend_v11`` (``kind`` in ``KINDS``, ``params`` of its final model)."""

    def __init__(self, name: str, kind: str, params: dict[str, Any]) -> None:
        super().__init__(name)
        if kind not in KINDS:
            raise ValueError(f"unknown v11 model kind {kind!r}")
        self._kind = kind
        self._params = dict(params)
        self._estimator: Any = None

    def clone(self) -> V11Member:
        return V11Member(self.name, self._kind, self._params)

    def get_params(self) -> dict[str, Any]:
        return {"kind": self._kind, "params": self._params}

    def _fit(self, x: pd.DataFrame, y: np.ndarray) -> None:
        self._estimator = pipeline(self._kind, self._params, list(x.columns)).fit(x, y)

    def _predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self._estimator.predict_proba(x)[:, 1]


class V11Blend(Model):
    """Fixed-weight logit average of v11 members (the original's ``ensemble.blend``)."""

    def __init__(self, name: str, members: Sequence[V11Member], weights: Sequence[float]) -> None:
        super().__init__(name)
        self._members = list(members)
        self._weights = list(weights)

    def clone(self) -> V11Blend:
        return V11Blend(self.name, [m.clone() for m in self._members], self._weights)

    def get_params(self) -> dict[str, Any]:
        return {m.name: w for m, w in zip(self._members, self._weights, strict=True)}

    def _fit(self, x: pd.DataFrame, y: np.ndarray) -> None:
        for member in self._members:
            member.fit(x, y)

    def _predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        P = np.stack([m.predict_proba(x) for m in self._members])
        w = np.asarray(self._weights)
        return expit(np.tensordot(w, logit(np.clip(P, 1e-6, 1 - 1e-6)), axes=1))
