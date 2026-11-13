"""Validation estimates of each team under three protocols, and simulated leaderboards.

    python docs/evaluation/comparison/stability.py --results results/comparison

``estimates.csv``: accuracy and log loss of each team's best private model under

* the team's own validation (ours: the seed-33 grouped folds; the 2nd team: the mean of its
  three partitions, ``second_place.py test``; predictive spikes: its notebook's
  ``StratifiedKFold``, with early stopping on the held-out fold);
* our folds, validated the way the team does (for predictive spikes: its training cell, early
  stopping on the held-out fold included);
* our folds with everything fitted on the training folds only, as for the real test set
  (``oof.csv`` of ``compare.py``; for us and the 2nd team this is the previous protocol; for
  predictive spikes the held-out fold is predicted as their test set, by the mean of five models
  early-stopped on folds of the training battles).

``split_simulation.csv``: the leaderboard drawn again many times from the out-of-fold
predictions: 5,000 battles without replacement, the first 1,500 "public" and the other 3,500
"private" (the competition's 30 / 70 split), accuracy of each team's model on each part (for
predictive spikes, of each of its three submissions).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

HERE = Path(__file__).resolve().parent
TEAMS = ("blend", "second", "spikes", "spikes_compact", "spikes_conservative")
# predictive spikes: name here -> column of their notebook's probabilities
SPIKES = {"spikes": "blend", "spikes_compact": "model4_compact_top140_te",
          "spikes_conservative": "model2_conservative_depth4_pruned_te"}  # fmt: skip
TEST, PUBLIC = 5000, 1500


def fold_mean(table: pd.DataFrame, column: str) -> dict[str, float]:
    """Accuracy and log loss of a column of probabilities, averaged over the folds."""
    groups: Any = table.groupby("fold")
    parts = []
    for _, group in groups:
        y, p = group["y_true"].to_numpy(), group[column].to_numpy()
        parts.append({"accuracy": accuracy_score(y, p >= 0.5), "log_loss": log_loss(y, p)})
    return {name: float(np.mean([part[name] for part in parts])) for name in parts[0]}


def second_own(results: Path) -> dict[str, float]:
    """The 2nd team's estimate of its best private model: mean of its three partitions."""
    own = pd.read_csv(results / "own_protocol" / "fixed" / "second_own_oof.csv")
    y = own["y_true"].to_numpy()
    columns = [c for c in own if c.startswith("tactical_team_spikes_")]
    parts = [{"accuracy": accuracy_score(y, own[c] >= 0.5), "log_loss": log_loss(y, own[c])}
             for c in columns]  # fmt: skip
    return {name: float(np.mean([part[name] for part in parts])) for name in parts[0]}


def estimates(oof: pd.DataFrame, results: Path) -> pd.DataFrame:
    """The three estimates of every model compared."""
    ours = fold_mean(oof, "blend")
    second = fold_mean(oof, "second")
    rows = [("blend", ours, ours, ours), ("second", second_own(results), second, second)]
    folder = results / "spikes" / "run"
    own_folds, our_folds = (pd.read_csv(folder / f"{name}_oof.csv") for name in ("own", "ours"))
    for model, column in SPIKES.items():
        own, cell = fold_mean(own_folds, column), fold_mean(our_folds, column)
        rows.append((model, own, cell, fold_mean(oof, model)))
    names = ("own validation", "our folds", "our folds, refitted")
    return pd.DataFrame(
        {"model": model, "protocol": name, **values}
        for model, *protocols in rows
        for name, values in zip(names, protocols, strict=True)
    )


def simulate(oof: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    """Accuracy of each team on the public and private part of resampled test sets."""
    y = oof["y_true"].to_numpy()
    right = {team: (oof[team].to_numpy() >= 0.5) == y for team in TEAMS}
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(resamples):
        battles = rng.choice(len(y), TEST, replace=False)
        public, private = battles[:PUBLIC], battles[PUBLIC:]
        for team, hits in right.items():
            rows.append({"draw": draw, "team": team, "public": hits[public].mean(),
                         "private": hits[private].mean()})  # fmt: skip
    return pd.DataFrame(rows)


def main(args: argparse.Namespace) -> None:
    oof = pd.read_csv(args.out / "oof.csv")
    table = estimates(oof, args.results)
    table.to_csv(args.out / "estimates.csv", index=False)
    print(table.to_string(float_format="{:.4f}".format))
    simulated = simulate(oof, args.resamples, args.seed)
    simulated.to_csv(args.out / "split_simulation.csv", index=False, float_format="%.5f")
    swing: Any = (simulated["private"] - simulated["public"]).groupby(simulated["team"])
    print(swing.agg(["mean", "std"]).to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="validation estimates, simulated leaderboards")
    parser.add_argument("--results", type=Path, required=True, help="results/comparison")
    parser.add_argument("--out", type=Path, default=HERE / "data")
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
