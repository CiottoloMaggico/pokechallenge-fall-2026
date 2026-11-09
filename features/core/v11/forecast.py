"""v11's learned forecast of the next turns.

LightGBM regressors, fitted on the training timelines, map the state after c entries (the usual
non-bag features recomputed on the truncated timeline) to the change of the faint difference, the
HP difference and each player's faints over the next h entries; applied to the last decision
point, they forecast how the battle goes on after it. Label-free: cross-fitted over the training
battles, while new battles only get predictions.

A copy of the original's ``features/forecast.py``, whose ``forecast_features`` is split into
``fit_forecast`` (out-of-fold training values and the regressors fitted on all truncated rows)
and ``predict_forecast``, so that new battles can be scored later with identical values.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

from features.core.v11.faint import N_JOBS, SEED
from features.core.v11.replay import Battle
from features.core.v11.state import p1_track, p2_track

CUTS = (12, 14, 16, 18, 20, 22, 24)
TARGETS = {10: ("fd", "hd", "f1", "f2"), 5: ("fd", "hd")}
INPUTS = ("s_", "f_", "t_", "a_", "e_", "m_", "w_", "x_", "d_", "v_", "z_", "c_")
PARAMS: dict[str, Any] = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.5,
    deterministic=True,
    force_row_wise=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    verbose=-1,
)

Regressors = dict[str, Any]


def inputs(columns: Iterable[str]) -> list[str]:
    """The regressors' input columns: the non-bag groups, then ``cut``."""
    return [c for c in columns if c.startswith(INPUTS)] + ["cut"]


def paths(battles: Sequence[Battle]) -> dict[str, np.ndarray]:
    """Per-entry trajectories: P1 and P2 faints (estimated hidden P2 faints included), their
    difference and the HP difference."""
    n = battles[0].n
    out = {k: np.zeros((len(battles), n)) for k in ("f1", "f2", "hd")}
    for k, b in enumerate(battles):
        t1, t2 = p1_track(b), p2_track(b)
        out["f1"][k], out["f2"][k] = t1["faint_cum"], t2["faint_cum"]
        out["hd"][k] = t1["hp_tot"] - t2["hp_tot"]
    out["fd"] = out["f2"] - out["f1"]
    return out


def _regressor() -> Any:
    return LGBMRegressor(**PARAMS)


def fit_forecast(
    train: Sequence[Battle], X_train: pd.DataFrame, truncated: pd.DataFrame
) -> tuple[pd.DataFrame, Regressors]:
    """Out-of-fold ``q_*`` columns of the training rows and the regressors fitted on all rows.

    ``train``: the full training battles (their detector already run); ``X_train``: their
    features indexed by battle id, with ``cut``; ``truncated``: their features at each cut.
    """
    cols = inputs(X_train.columns)
    traj = paths(train)
    ids = X_train.index.to_numpy()
    row_of = {bid: k for k, bid in enumerate(ids)}
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(ids))
    out_train: dict[str, np.ndarray] = {}
    regressors: Regressors = {}
    for h, targets in TARGETS.items():
        Z: Any = truncated[truncated["cut"] + h <= traj["fd"].shape[1]]
        bid = Z.index.to_numpy()
        rows = np.array([row_of[b] for b in bid])
        start = Z["cut"].to_numpy().astype(int) - 1
        Z = Z[cols]
        for t in targets:
            y = traj[t][rows, start + h] - traj[t][rows, start]
            oof = np.zeros(len(ids))
            for tr, va in folds:
                m = np.isin(bid, ids[tr])
                oof[va] = _regressor().fit(Z[m], y[m]).predict(X_train.iloc[va][cols])
            out_train[f"q_{t}_h{h}"] = oof
            regressors[f"q_{t}_h{h}"] = _regressor().fit(Z, y)
    return pd.DataFrame(out_train, index=X_train.index), regressors


def predict_forecast(regressors: Regressors, X_new: pd.DataFrame) -> pd.DataFrame:
    """The ``q_*`` columns of new battles (``X_new`` with ``cut``, as the training matrix)."""
    cols = inputs(X_new.columns)
    out = {name: model.predict(X_new[cols]) for name, model in regressors.items()}
    return pd.DataFrame(out, index=X_new.index)
