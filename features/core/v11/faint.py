"""v11's hidden-faint detector.

Player 2 faints are only recorded when they coincide with a Player 1 faint, so a model trained on
Player 1 transitions, where faints are known, scores every Player 2 disappearance. A copy of the
original's ``faint.py`` (LightGBM, ``SEED = 42``, ``N_JOBS = 4`` threads).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from features.core.v11.replay import Battle

SEED = 42
N_JOBS = 4
META = ("b", "i", "x", "recorded", "final", "label")
PARAMS: dict[str, Any] = dict(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    deterministic=True,
    force_row_wise=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    verbose=-1,
)


def _collect(battles: Sequence[Battle], side: str) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    for k, b in enumerate(battles):
        for r in b.transitions(side):
            r["b"] = k
            rows.append(r)
    df = pd.DataFrame(rows)
    feats = [c for c in df.columns if c not in META]
    df[feats] = df[feats].astype(float)
    return df, feats


def _fit(df: pd.DataFrame, feats: list[str]) -> Any:
    return lgb.LGBMClassifier(**PARAMS).fit(cast(pd.DataFrame, df[feats]), df["label"])


def fit_detector(
    train: Sequence[Battle], test: Sequence[Battle], n_splits: int = 5
) -> dict[str, float]:
    """Learn P(faint | disappearance) on p1 transitions and apply it to p2 disappearances.

    Labels are observed on p1. The training battles get cross-fitted probabilities (folds of
    battles), the test battles those of a model fitted on all p1 rows; recorded p2 faints get 1.
    Writes ``battle.p2_faint`` and returns the detector's out-of-fold diagnostics.
    """
    p1, feats = _collect(train, "p1")
    p1 = cast(pd.DataFrame, p1[p1["final"] & p1["x_hp"].notna()])
    p2_tr, _ = _collect(train, "p2")

    fold_of = np.empty(len(train), dtype=int)
    folds = KFold(n_splits, shuffle=True, random_state=SEED).split(np.arange(len(train)))
    for f, (_, idx) in enumerate(folds):
        fold_of[idx] = f
    oof_p1 = np.zeros(len(p1))
    p2_tr["p"] = 0.0
    cand_tr = (~p2_tr["recorded"]) & p2_tr["final"]
    for f in range(n_splits):
        in_f = fold_of[p1["b"].to_numpy()] == f
        model = _fit(cast(pd.DataFrame, p1[~in_f]), feats)
        oof_p1[in_f] = model.predict_proba(p1.loc[in_f, feats])[:, 1]
        sel = cand_tr & (fold_of[p2_tr["b"].to_numpy()] == f)
        p2_tr.loc[sel, "p"] = model.predict_proba(p2_tr.loc[sel, feats])[:, 1]
    frames = [(p2_tr, train)]
    if test:
        p2_te, _ = _collect(test, "p2")
        full = _fit(p1, feats)
        p2_te["p"] = 0.0
        cand_te = (~p2_te["recorded"]) & p2_te["final"]
        p2_te.loc[cand_te, "p"] = full.predict_proba(p2_te.loc[cand_te, feats])[:, 1]
        frames.append((p2_te, test))
    for df, battles in frames:
        df.loc[df["recorded"], "p"] = 1.0
        for b, i, p in zip(df["b"], df["i"], df["p"], strict=True):
            battles[b].p2_faint[int(i)] = float(p)
    pos_rate: Any = p1["label"].mean()
    hidden: Any = p2_tr.loc[cand_tr, "p"].sum() / len(train)
    return {
        "detector_auc": roc_auc_score(p1["label"], oof_p1),
        "detector_pos_rate": float(pos_rate),
        "p2_hidden_per_battle": float(hidden),
    }
