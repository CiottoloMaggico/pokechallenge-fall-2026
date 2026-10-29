"""Greedy feature selection on repeated, grouped cross-validation.

Starting from the columns of one feature set (``--start``), columns of a
larger pool (``--pool``) are added or removed one at a time, comparing
each candidate with the current set fold by fold (paired):

* add the candidate that lowers the log-loss the most, if its paired
  t-statistic is at most ``--add-t`` (default -2.5) and the accuracy
  does not drop;
* otherwise remove the column whose removal keeps the accuracy and does
  not worsen the log-loss significantly (t below ``--drop-t``, 1.0);
* stop when nothing changes (or a step would return to an earlier set).

The pool's tables are built once (stateful features fitted inside each
training part, as in ``gym``), so each comparison only refits the
model. Battles are repaired with the knockout detector, exactly as in
``main.py``.

Example::

    python selection.py --test input/test.jsonl
    python selection.py --test input/test.jsonl --model logreg --repeats 3

The result (steps, final columns, scores) is printed and saved to
``--out`` (default ``runs/selection.json``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

import feature_catalog
import model_catalog
from data import Dataset
from domain.pokedex import Pokedex
from gym import FoldData, prepare_folds
from knockout_detector import detector_from_files

DEFAULT_POKEDEX = Path("assets/species.json")
DEFAULT_TRAIN = Path("input/train.jsonl")
DEFAULT_OUT = Path("runs/selection.json")
PROBA_CLIP = 1e-6

Scores = tuple[np.ndarray, np.ndarray]  # accuracy and log-loss per fold


class RepeatedGroupFolds:
    """``StratifiedGroupKFold`` repeated with several seeds."""

    def __init__(self, n_splits: int, seeds: Sequence[int]) -> None:
        self.n_splits = n_splits
        self.seeds = tuple(seeds)

    def split(
        self, x: np.ndarray, y: np.ndarray, groups: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for seed in self.seeds:
            folds = StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=seed)
            yield from folds.split(x, y, groups)


class Selector:
    """Scores column subsets on precomputed fold tables."""

    def __init__(self, folds: Sequence[FoldData], model: str) -> None:
        self.folds = folds
        self.model = model

    @property
    def columns(self) -> list[str]:
        return list(self.folds[0][0].columns)

    def score(self, columns: Sequence[str]) -> Scores:
        accuracy, loss = [], []
        for x_train, y_train, x_valid, y_valid in self.folds:
            model = model_catalog.build(self.model)
            model.fit(x_train[list(columns)], y_train)
            proba = np.clip(
                model.predict_proba(x_valid[list(columns)]), PROBA_CLIP, 1 - PROBA_CLIP
            )
            accuracy.append(accuracy_score(y_valid, proba > 0.5))
            loss.append(log_loss(y_valid, proba))
        return np.array(accuracy), np.array(loss)


def compare(new: Scores, old: Scores) -> tuple[float, float, float]:
    """Mean accuracy change, mean log-loss change, paired t of the log-loss."""
    d_accuracy, d_loss = new[0] - old[0], new[1] - old[1]
    t = float(stats.ttest_1samp(d_loss, 0).statistic) if d_loss.std() > 0 else 0.0
    return float(d_accuracy.mean()), float(d_loss.mean()), t


def select(
    selector: Selector, start: Sequence[str], add_t: float, drop_t: float
) -> tuple[list[str], list[dict[str, object]], Scores]:
    current = list(start)
    scores = selector.score(current)
    steps: list[dict[str, object]] = []
    visited = {frozenset(current)}
    _report("start", "", scores, current)
    while True:
        best = None
        for column in selector.columns:
            if column in current:
                continue
            trial = selector.score([*current, column])
            d_acc, d_loss, t = compare(trial, scores)
            if t <= add_t and d_acc >= 0 and (best is None or d_loss < best[2]):
                best = (column, trial, d_loss, d_acc, t)
        action = "add"
        if best is None:
            action = "drop"
            for column in current:
                trial = selector.score([c for c in current if c != column])
                d_acc, d_loss, t = compare(trial, scores)
                if t < drop_t and d_acc >= 0 and (best is None or d_acc > best[3]):
                    best = (column, trial, d_loss, d_acc, t)
        if best is None:
            return current, steps, scores
        column, trial_scores, d_loss, d_acc, t = best
        changed = (
            [*current, column]
            if action == "add"
            else [c for c in current if c != column]
        )
        if frozenset(changed) in visited:  # would go back to an earlier set
            return current, steps, scores
        visited.add(frozenset(changed))
        current, scores = changed, trial_scores
        steps.append(
            {
                "action": action,
                "column": column,
                "accuracy_change": d_acc,
                "log_loss_change": d_loss,
                "t": t,
            }
        )
        _report(action, column, scores, current)


def _report(action: str, column: str, scores: Scores, current: Sequence[str]) -> None:
    print(
        f"{action:5s} {column:24s} accuracy {scores[0].mean():.4f}  "
        f"log-loss {scores[1].mean():.4f}  ({len(current)} columns)",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, help="unlabelled battles (optional)")
    parser.add_argument("--pokedex", type=Path, default=DEFAULT_POKEDEX)
    parser.add_argument("--pool", default="selection_pool")
    parser.add_argument("--start", default="compact_v3")
    parser.add_argument("--model", default="logreg")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--add-t", type=float, default=-2.5)
    parser.add_argument("--drop-t", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pokedex = Pokedex.from_json(args.pokedex)
    chart = feature_catalog.type_chart()
    detector = detector_from_files(pokedex, chart, (args.train, args.test))
    train = Dataset.from_jsonl(args.train, pokedex, repair=detector)
    unlabeled = (
        Dataset.from_jsonl(args.test, pokedex, repair=detector).contexts
        if args.test
        else ()
    )
    start = feature_catalog.build(args.start).declared_columns
    if start is None:
        raise SystemExit(f"{args.start!r} does not declare its columns")
    seeds = [args.seed + 11 * k for k in range(args.repeats)]
    began = time.perf_counter()
    folds = prepare_folds(
        feature_catalog.build(args.pool),
        train.contexts,
        train.labels,
        RepeatedGroupFolds(args.folds, seeds),
        train.duplicate_groups(),
        unlabeled,
        progress=lambda done, total, s: print(f"  fold {done}/{total} ({s:.0f} s)"),
    )
    print(f"pool tables ready in {time.perf_counter() - began:.0f} s")
    selector = Selector(folds, args.model)
    missing = set(start) - set(selector.columns)
    if missing:
        raise SystemExit(f"start columns not in the pool: {sorted(missing)}")
    columns, steps, scores = select(selector, start, args.add_t, args.drop_t)
    result = {
        "start": args.start,
        "pool": args.pool,
        "model": args.model,
        "folds": len(folds),
        "columns": columns,
        "accuracy": float(scores[0].mean()),
        "log_loss": float(scores[1].mean()),
        "steps": steps,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"final columns ({len(columns)}): {columns}")
    print(f"saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
