"""The tables of the comparison with two other teams, from every team's predictions.

    python docs/evaluation/comparison/compare.py --results results/comparison

Reads our out-of-fold probabilities (``docs/evaluation/data/oof.csv``, seed-33 folds) and, under
``--results``, the other teams' models rebuilt on the same folds (``second_place.py``,
``predictive_spikes.py``; the layout is in the README). Writes to ``data/``:

* ``oof.csv``: every model's out-of-fold probability per battle;
* ``folds.csv``, ``metrics.csv``: metrics per fold and their mean;
* ``paired.csv``: paired differences per fold, corrected repeated k-fold t-test;
* ``loss_curve.csv``: validation log loss on fold 0 by number of training battles;
* ``agreement.csv``: share of battles where two models predict the same winner (out of fold
  and on the test set).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from stats_tables import corrected_ttest  # noqa: E402

OURS = ["compact_v5", "v11", "blend"]
# the 2nd-placed team: column of second_oof.csv / second_test.csv -> name here
SECOND = {"team_similarity": "second_teams", "tactical_team_spikes": "second"}
# predictive spikes (9th): column of their notebook's probabilities -> name here
SPIKES = {
    "model4_compact_top140_te": "spikes_compact",
    "model2_conservative_depth4_pruned_te": "spikes_conservative",
    "blend": "spikes",
}
MODELS = [*OURS, "second_five_views", *SECOND.values(), *SPIKES.values()]
PAIRS = [("blend", "second"), ("blend", "spikes"), ("second", "spikes"), ("compact_v5", "second"),
         ("v11", "second"), ("compact_v5", "spikes"), ("v11", "spikes")]  # fmt: skip
METRICS = {
    "accuracy": lambda y, p: accuracy_score(y, p >= 0.5),
    "log_loss": lambda y, p: log_loss(y, p, labels=[0, 1]),
    "roc_auc": roc_auc_score,
    "brier": brier_score_loss,
}
FIVE_VIEWS = ["baseline", "species_forecast", "tactical", "zero_power", "raw_summary"]


def add(table: pd.DataFrame, part: pd.DataFrame, columns: dict[str, Any]) -> None:
    """Add models to ``table`` after checking that ``part`` has the same battles and folds."""
    part = part.sort_values("row").reset_index(drop=True)
    same = all(part[key].equals(table[key]) for key in ("row", "fold", "y_true"))
    if not same:
        raise SystemExit("predictions that do not share our folds and labels")
    for name, values in columns.items():
        table[name] = np.asarray(values(part))


def oof_table(results: Path) -> pd.DataFrame:
    """Every model's out-of-fold probability on our folds, one row per training battle."""
    table = pd.read_csv(HERE.parent / "data" / "oof.csv").sort_values("row").reset_index(drop=True)
    original = pd.read_csv(results / "our_folds" / "original" / "components.csv")
    add(table, original, {"second_five_views": lambda part: part[FIVE_VIEWS].mean(axis=1)})
    fixed = pd.read_csv(results / "our_folds" / "fixed" / "second_oof.csv")
    add(table, fixed, {name: lambda part, c=column: part[c] for column, name in SECOND.items()})
    refitted = pd.read_csv(results / "spikes" / "deploy" / "deploy_oof.csv")
    add(table, refitted, {name: lambda part, c=column: part[c] for column, name in SPIKES.items()})
    return table.sort_values(["fold", "row"]).reset_index(drop=True)


def by_fold(oof: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Every metric of every model on every fold (long format)."""
    rows = []
    groups: Any = oof.groupby("fold")
    for fold, group in groups:
        y = group["y_true"].to_numpy()
        for model in models:
            p = group[model].to_numpy()
            scores = {name: float(fn(y, p)) for name, fn in METRICS.items()}
            rows.append({"fold": fold, "model": model, **scores})
    return pd.DataFrame(rows)


def paired_table(folds: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """Mean paired difference a - b per metric with the corrected t-test's p-value."""
    rows = []
    for a, b in PAIRS:
        merged = folds[folds["model"] == a].merge(folds[folds["model"] == b], on="fold")
        for name in METRICS:
            diff = (merged[f"{name}_x"] - merged[f"{name}_y"]).to_numpy()
            p = corrected_ttest(diff, 1 / (n_folds - 1))[1]
            rows.append({"a": a, "b": b, "metric": name, "delta": diff.mean(), "p": p})
    return pd.DataFrame(rows)


def curve_table(results: Path) -> pd.DataFrame:
    """Validation log loss on fold 0 by training size: ``blend``, the 2nd team's tactical view
    (its second stage needs out-of-fold rows of every battle, so it has no curve) and
    predictive spikes' blend."""
    ours: Any = pd.read_csv(HERE.parent / "data" / "loss_curve.csv")
    ours = ours[(ours["split"] == "valid") & (ours["model"] == "blend")]
    sources = {
        "second": (results / "curve" / "curve.csv", "tactical"),
        "spikes": (results / "spikes" / "curve" / "curve.csv", "p"),
    }
    rows = []
    for model, (path, column) in sources.items():
        groups: Any = pd.read_csv(path).groupby("size")
        for size, part in groups:
            loss = float(log_loss(part["y_true"], part[column], labels=[0, 1]))
            rows.append({"model": model, "split": "valid", "size": size, "fold": 0,
                         "n_train": int(part["n_train"].iloc[0]), "log_loss": loss})  # fmt: skip
    return pd.concat([ours, pd.DataFrame(rows)], ignore_index=True)


def test_table(results: Path) -> pd.DataFrame:
    """The test probabilities of every model rebuilt here, keyed by battle id."""
    table = pd.DataFrame()
    for model in OURS:
        proba = pd.read_csv(results / "ours" / f"proba_{model}.csv").set_index("battle_id")
        table[model] = proba["p_player_won"]
    theirs = pd.read_csv(results / "own_protocol" / "fixed" / "second_test.csv")
    theirs = theirs.set_index("battle_id")
    for column, name in SECOND.items():
        table[name] = theirs[column]
    proba = pd.read_csv(results / "spikes" / "run" / "test_proba.csv").set_index("battle_id")
    for column, name in SPIKES.items():
        table[name] = proba[column]
    return table


def agreement(oof: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Share of battles with the same predicted winner, out of fold and on the test set."""
    pairs = [("compact_v5", "v11"), ("blend", "second"), ("blend", "spikes"), ("second", "spikes")]
    rows = []
    for a, b in pairs:
        same = [float(np.mean((t[a] >= 0.5).to_numpy() == (t[b] >= 0.5).to_numpy()))
                for t in (oof, test)]  # fmt: skip
        rows.append({"a": a, "b": b, "same_oof": same[0], "same_test": same[1]})
    return pd.DataFrame(rows)


def main(results: Path, out: Path) -> None:
    oof = oof_table(results)
    folds = by_fold(oof, MODELS)
    grouped: Any = folds.groupby("model")
    metrics = grouped[list(METRICS)].mean().loc[MODELS]
    metrics["accuracy_fold_std"] = grouped["accuracy"].std()
    test = test_table(results)
    out.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out / "oof.csv", index=False, float_format="%.6f")
    folds.to_csv(out / "folds.csv", index=False)
    metrics.reset_index().to_csv(out / "metrics.csv", index=False)
    paired_table(folds, len(set(oof["fold"]))).to_csv(out / "paired.csv", index=False)
    curve_table(results).to_csv(out / "loss_curve.csv", index=False)
    agreement(oof, test).to_csv(out / "agreement.csv", index=False)
    print(metrics.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tables of the comparison with other teams")
    parser.add_argument("--results", type=Path, required=True, help="results/comparison")
    parser.add_argument("--out", type=Path, default=HERE / "data")
    arguments = parser.parse_args()
    main(arguments.results, arguments.out)
