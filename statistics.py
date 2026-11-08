"""Tables and tests derived from the raw results of the gym.

Every comparison printed by the program uses :func:`corrected_ttest`. The functions read the
``folds`` and ``predictions`` tables only, so they also work on results loaded back from disk.
"""
# pandas ships no complete types: strict mode checks our own signatures and logic, not its calls.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
from scipy import stats  # pyright: ignore[reportMissingTypeStubs]

SEED_STD_METRICS = ("accuracy", "log_loss")  # metrics that also get a seed_std


def corrected_ttest(
    diffs: Sequence[float] | np.ndarray, test_train_ratio: float
) -> tuple[float, float]:
    """Corrected repeated k-fold t-test (Nadeau & Bengio 2003; Bouckaert &
    Frank 2004) of paired per-fold differences.

    The J differences of r repetitions of k-fold CV are not independent
    (training sets overlap and the repetitions reuse the same battles), so
    the variance of their mean is ``(1/J + n_test/n_train) * s^2`` instead
    of ``s^2 / J``: the naive t is divided by ``sqrt(1 + J * n_test/n_train)``.
    Note the floor: more seeds shrink 1/J, never the n_test/n_train term.
    Returns ``(t, two-sided p)`` with J - 1 degrees of freedom.
    """
    d = np.asarray(diffs, dtype=float)
    j = d.size
    if j < 2:
        return float("nan"), float("nan")
    var = d.var(ddof=1)
    if var == 0:
        return (0.0, 1.0) if d.mean() == 0 else (float(np.sign(d.mean()) * np.inf), 0.0)
    t = d.mean() / np.sqrt((1 / j + test_train_ratio) * var)
    return float(t), float(2 * stats.t.sf(abs(t), j - 1))


def summary_table(folds: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Mean and std of each metric per (feature set, model).

    With one seed (no ``seed`` column, or a single value) the columns are
    ``<metric>_mean`` and ``<metric>_std`` over the folds. With several
    seeds of the fold assignment (``main.py evaluate --n-seeds``):

    * ``<metric>_mean``: mean over every seed x fold;
    * ``<metric>_std``: fold-to-fold std within a seed, averaged over the
      seeds (same meaning as with one seed);
    * ``<metric>_seed_std``: std across seeds of the per-seed means, i.e.
      how much the number moves with the luck of the fold assignment
      (accuracy and log loss only, to keep the table short).
    """
    if "seed" not in folds or len(set(folds["seed"])) < 2:
        aggregations = {
            f"{metric}_{stat}": (metric, stat) for metric in metrics for stat in ("mean", "std")
        }
        grouped = folds.groupby(["feature_set", "model"])
        return grouped.agg(**aggregations).reset_index()
    by_seed: Any = folds.groupby(["feature_set", "model", "seed"])[list(metrics)]
    means, stds = by_seed.mean(), by_seed.std()
    keys = ["feature_set", "model"]
    columns: dict[str, pd.Series] = {}
    for metric in metrics:
        columns[f"{metric}_mean"] = means[metric].groupby(level=keys).mean()
        columns[f"{metric}_std"] = stds[metric].groupby(level=keys).mean()
        if metric in SEED_STD_METRICS:
            columns[f"{metric}_seed_std"] = means[metric].groupby(level=keys).std()
    table = pd.DataFrame(columns).reset_index()
    table.insert(2, "n_seeds", means.groupby(level=keys).size().to_numpy())
    return table


def seed_table(folds: pd.DataFrame, metric: str = "accuracy") -> pd.DataFrame:
    """Mean ``metric`` over the folds of each seed: one row per (feature
    set, model), one column per seed."""
    if "seed" not in folds:
        raise KeyError("these results have a single seed")
    table = folds.pivot_table(
        index=["feature_set", "model"], columns="seed", values=metric, aggfunc="mean"
    )
    table.columns = [f"seed {s}" for s in table.columns]
    return table.reset_index()


def paired_table(
    folds: pd.DataFrame, reference: tuple[str, str], metric: str = "accuracy"
) -> pd.DataFrame:
    """Per-fold ``metric`` difference of each combination vs reference.

    Columns: feature_set, model, mean_diff, std_diff, folds_better (seed x
    fold pairs where the combination is better), and ``t`` / ``p`` of the
    corrected repeated k-fold t-test (:func:`corrected_ttest`), which does
    not treat overlapping folds and repeated seeds as independent.
    """
    is_ref = _mask(folds, reference[0], reference[1])
    if not bool(is_ref.any()):
        raise KeyError(f"reference {reference} not in the results")
    keys = ["seed", "fold"] if "seed" in folds else ["fold"]
    ref = folds.loc[is_ref, [*keys, metric]].rename(columns={metric: "reference"})
    merged = folds.merge(ref, on=keys)
    merged["diff"] = merged[metric] - merged["reference"]
    merged["better"] = (merged["diff"] > 0).astype(int)
    ratios: Any = folds["n_valid"] / folds["n_train"]
    ratio = float(ratios.mean())
    table = merged.groupby(["feature_set", "model"]).agg(
        mean_diff=("diff", "mean"),
        std_diff=("diff", "std"),
        folds_better=("better", "sum"),
    )
    tests = merged.groupby(["feature_set", "model"])["diff"].apply(
        lambda d: corrected_ttest(d.to_numpy(), ratio)
    )
    table["t"] = [t for t, _ in tests.loc[table.index]]
    table["p"] = [p for _, p in tests.loc[table.index]]
    return table.reset_index().sort_values("mean_diff", ascending=False)


def calibration_table(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed win rate per probability bin."""
    preds = predictions.copy()
    preds["bin"] = np.minimum((preds["proba"] * bins).astype(int), bins - 1)
    grouped = preds.groupby(["feature_set", "model", "bin"])
    table = grouped.agg(
        mean_proba=("proba", "mean"),
        observed=("y_true", "mean"),
        count=("y_true", "size"),
    )
    return table.reset_index()


def threshold_table(
    predictions: pd.DataFrame, thresholds: Sequence[float] | None = None
) -> pd.DataFrame:
    """Accuracy as a function of the decision threshold."""
    grid = np.linspace(0.3, 0.7, 41) if thresholds is None else thresholds
    combos = predictions[["feature_set", "model"]].drop_duplicates()
    rows: list[dict[str, Any]] = []
    for fs, model in combos.itertuples(index=False, name=None):
        group = predictions[_mask(predictions, fs, model)]
        y, p = np.asarray(group["y_true"]), np.asarray(group["proba"])
        for t in grid:
            rows.append(
                {
                    "feature_set": fs,
                    "model": model,
                    "threshold": float(t),
                    "accuracy": float(((p > t).astype(int) == y).mean()),
                }
            )
    return pd.DataFrame(rows)


def _mask(table: pd.DataFrame, feature_set: str, model: str) -> np.ndarray:
    """Boolean row mask for one (feature set, model) combination."""
    fs_match = np.asarray(table["feature_set"]) == feature_set
    model_match = np.asarray(table["model"]) == model
    return fs_match & model_match
