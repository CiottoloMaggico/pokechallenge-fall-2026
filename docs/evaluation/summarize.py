"""The small tables behind the notebook's evaluation, from the outputs of the experiments.

    python docs/evaluation/summarize.py models --compact <evaluation> --v11 <evaluation> --out D
    python docs/evaluation/summarize.py turns --runs <folder of t<N>/> --full <evaluation> --out D
    python docs/evaluation/summarize.py curve --compact <csv> --v11 <csv> [<csv> ...] --out D

``models``: out-of-fold probabilities of the three submitted models on the same folds
(``oof.csv``), their metrics per model (``metrics.csv``) and the paired differences with the
corrected repeated k-fold t-test (``paired.csv``). ``turns``: the folds of compact_v5 on battles
cut to their first N turns (``accuracy_by_turn.csv``). ``curve``: training and validation log
loss by training size, from ``learning_curve.py`` (``loss_curve.csv``). The blend is always
computed here from the two models' probabilities (``blend.py``).
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from blend import blend  # noqa: E402

from stats_tables import corrected_ttest  # noqa: E402

COMPACT, V11, BLEND = "compact_v5", "v11", "blend"
MODELS = (COMPACT, V11, BLEND)
METRICS = {
    "accuracy": lambda y, p: accuracy_score(y, p >= 0.5),
    "log_loss": lambda y, p: log_loss(y, p, labels=[0, 1]),
    "roc_auc": roc_auc_score,
    "brier": brier_score_loss,
}
KEYS = ["seed", "fold", "row"]


def predictions(run: Path, feature_set: str, model: str) -> pd.DataFrame:
    """The out-of-fold predictions of one combination of an ``evaluate`` folder."""
    table = pd.read_csv(run / "predictions.csv")
    table = table[(table["feature_set"] == feature_set) & (table["model"] == model)]
    if table.empty:
        raise SystemExit(f"{run}: no predictions of {feature_set} / {model}")
    return table.loc[:, [*KEYS, "y_true", "proba"]]


def by_fold(oof: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Every metric of every model on every group of ``keys`` (long format)."""
    rows = []
    groups: Any = oof.groupby(keys)
    for key, group in groups:
        y = group["y_true"].to_numpy()
        for model in MODELS:
            scores = {name: float(fn(y, group[model].to_numpy())) for name, fn in METRICS.items()}
            rows.append({**dict(zip(keys, key, strict=True)), "model": model, **scores})
    return pd.DataFrame(rows)


def models(args: argparse.Namespace) -> None:
    compact = predictions(args.compact, COMPACT, args.compact_model)
    v11 = predictions(args.v11, V11, args.v11_model)
    oof: Any = compact.merge(v11, on=KEYS, suffixes=("", "_v11"), validate="1:1")
    if len(oof) != len(compact) or not oof["y_true"].equals(oof["y_true_v11"]):
        raise SystemExit("the two evaluations do not share their folds: use the same --seeds")
    oof = oof.rename(columns={"proba": COMPACT, "proba_v11": V11}).drop(columns="y_true_v11")
    if args.seeds:
        oof = oof[oof["seed"].isin(args.seeds)]
    oof[BLEND] = blend(oof[COMPACT].to_numpy(), oof[V11].to_numpy())
    folds = by_fold(oof, ["seed", "fold"])
    names = list(METRICS)
    grouped: Any = folds.groupby("model")
    table = grouped[names].mean().loc[list(MODELS)]
    table["accuracy_fold_std"] = grouped["accuracy"].std()
    seeds: Any = folds.groupby(["model", "seed"])["accuracy"].mean()
    table["accuracy_seed_std"] = seeds.groupby("model").std()
    table["folds"] = grouped.size()
    ratio = 1 / (len(set(oof["fold"])) - 1)  # validation / training battles of a k-fold
    paired = []
    for a, b in combinations(MODELS, 2):
        merged = folds[folds["model"] == a].merge(folds[folds["model"] == b], on=["seed", "fold"])
        for name in names:
            diff = (merged[f"{name}_x"] - merged[f"{name}_y"]).to_numpy()
            p = corrected_ttest(diff, ratio)[1]
            paired.append({"a": a, "b": b, "metric": name, "delta": diff.mean(), "p": p})
    args.out.mkdir(parents=True, exist_ok=True)
    oof.to_csv(args.out / "oof.csv", index=False, float_format="%.6f")
    table.reset_index().to_csv(args.out / "metrics.csv", index=False)
    pd.DataFrame(paired).to_csv(args.out / "paired.csv", index=False)
    print(table.to_string(float_format="{:.4f}".format))


def turns(args: argparse.Namespace) -> None:
    names = ["accuracy", "log_loss", "roc_auc", "brier"]
    parts = []
    for folder in sorted(args.runs.glob("t[0-9]*/"), key=lambda path: int(path.name[1:])):
        parts.append(pd.read_csv(folder / "folds.csv").assign(turn=int(folder.name[1:])))
    full = pd.read_csv(args.full / "folds.csv")
    full = full[(full["model"] == args.model) & (full["seed"] == args.seed)]
    parts.append(full.assign(turn=args.last_turn))
    table = pd.concat(parts, ignore_index=True)
    table = table[table["model"] == args.model].loc[:, ["turn", "fold", *names]]
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "accuracy_by_turn.csv", index=False)
    shown: Any = table.groupby("turn")[names].mean()
    print(shown.to_string(float_format="{:.4f}".format))


def curve(args: argparse.Namespace) -> None:
    keys = ["fold", "size", "split", "row"]
    compact = pd.read_csv(args.compact).loc[:, [*keys, "n_train", "y_true", "proba"]]
    v11 = pd.concat([pd.read_csv(path) for path in args.v11]).loc[:, [*keys, "proba"]]
    both = compact.merge(v11, on=keys, suffixes=("", "_v11"), validate="1:1")
    both = both.rename(columns={"proba": COMPACT, "proba_v11": V11})
    both[BLEND] = blend(both[COMPACT].to_numpy(), both[V11].to_numpy())
    rows = []
    groups: Any = both.groupby(["fold", "size", "split"])
    for (fold, size, split), group in groups:
        y, n_train = group["y_true"].to_numpy(), int(group["n_train"].iloc[0])
        for model in MODELS:
            loss = float(log_loss(y, group[model].to_numpy(), labels=[0, 1]))
            rows.append({"model": model, "split": split, "size": size, "fold": fold,
                         "n_train": n_train, "log_loss": loss})  # fmt: skip
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "loss_curve.csv", index=False)
    shown: Any = table.groupby(["size", "model", "split"])["log_loss"].mean()
    print(shown.unstack(["model", "split"]).to_string(float_format="{:.4f}".format))


def parser() -> argparse.ArgumentParser:
    """One sub-command per table family."""
    root = argparse.ArgumentParser(description="tables of the notebook's evaluation")
    commands = root.add_subparsers(dest="command", required=True)
    p_models = commands.add_parser("models", help="oof.csv, metrics.csv, paired.csv")
    p_models.add_argument("--compact", type=Path, required=True, help="evaluation of compact_v5")
    p_models.add_argument("--v11", type=Path, required=True, help="evaluation of v11")
    p_models.add_argument("--compact-model", default="ensemble_from_lr-tuned")
    p_models.add_argument("--v11-model", default="v11_blend")
    p_models.add_argument("--seeds", type=int, nargs="+", help="keep these seeds (default: all)")
    p_models.set_defaults(handler=models)
    p_turns = commands.add_parser("turns", help="accuracy_by_turn.csv")
    p_turns.add_argument("--runs", type=Path, required=True, help="folder of t<N> evaluations")
    p_turns.add_argument("--full", type=Path, required=True, help="evaluation on whole battles")
    p_turns.add_argument("--model", default="ensemble_from_lr-tuned")
    p_turns.add_argument("--seed", type=int, default=33)
    p_turns.add_argument("--last-turn", type=int, default=30)
    p_turns.set_defaults(handler=turns)
    p_curve = commands.add_parser("curve", help="loss_curve.csv")
    p_curve.add_argument("--compact", type=Path, required=True)
    p_curve.add_argument("--v11", type=Path, nargs="+", required=True)
    p_curve.set_defaults(handler=curve)
    for sub in (p_models, p_turns, p_curve):
        sub.add_argument("--out", type=Path, required=True, help="folder of the tables")
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.handler(arguments)
