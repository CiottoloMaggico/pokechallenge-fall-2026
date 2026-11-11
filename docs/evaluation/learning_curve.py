"""Learning curve: predictions of one feature set's models trained on growing parts of a fold.

    python docs/evaluation/learning_curve.py --features compact_v5 --models ensemble_from_lr \\
        --params assets/params/compact_v5__*_lr.json --out results/evaluation/curve_compact_v5.csv
    python docs/evaluation/learning_curve.py --features v11 --models v11_blend \\
        --repair v11=results/addons/knockouts_v11/v1 --out results/evaluation/curve_v11.csv

For every fold of the plan (the folds of ``evaluate`` with the same seed) and every size, the
feature set and the models are fitted on the first ``size`` training battles of a fixed random
order (whole duplicate groups, so the subsets are nested and never leak into the validation
fold), then predict those battles ("train") and the validation fold ("valid"). Rows are appended
to ``--out``; a (fold, size) already there is skipped, so the script can be stopped and resumed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from addons import load_repair  # noqa: E402
from data import Dataset  # noqa: E402
from domain.pokedex import Pokedex  # noqa: E402
from features import sets as feature_sets  # noqa: E402
from folds import FoldPlan, make_splits  # noqa: E402
from interfaces import LabelledBattles, Model  # noqa: E402
from settings import DEFAULT_POKEDEX, DEFAULT_REPAIR, DEFAULT_TRAIN  # noqa: E402
from usecases import build_models  # noqa: E402

SIZES = tuple(range(250, 8000, 250))
ORDER_SEED = 0


def nested_subset(train: np.ndarray, groups: np.ndarray, size: int) -> np.ndarray:
    """The first ``size`` rows of ``train`` in a fixed random order of its duplicate groups
    (a group is never split, so the subset can exceed ``size`` by a few duplicates)."""
    order = np.random.default_rng(ORDER_SEED).permutation(np.unique(groups[train]))
    rank = {group: k for k, group in enumerate(order.tolist())}
    ranked = sorted(train.tolist(), key=lambda row: rank[groups[row]])
    last = rank[groups[ranked[min(size, len(ranked)) - 1]]]
    return np.array([row for row in ranked if rank[groups[row]] <= last])


def done(out: Path) -> set[tuple[int, int]]:
    """The (fold, size) pairs already written to ``out``."""
    if not out.exists():
        return set()
    old = pd.read_csv(out, usecols=["fold", "size"]).drop_duplicates()
    return set(map(tuple, old.to_numpy().tolist()))


def predictions(data: LabelledBattles, source: str, models: list[Model], rows: np.ndarray,
                valid: np.ndarray) -> pd.DataFrame:  # fmt: skip
    """Fit the feature set ``source`` and each model on ``rows``; predict ``rows`` and ``valid``."""
    features = feature_sets.build(source)
    x_train = features.fit_transform(
        [data.items[i] for i in rows], data.labels[rows].tolist(), (), data.groups[rows].tolist()
    )
    x_valid = features.to_frame([data.items[i] for i in valid])
    parts = []
    for model in models:
        fitted = model.clone().fit(x_train, data.labels[rows])
        for split, index, x in (("train", rows, x_train), ("valid", valid, x_valid)):
            part = pd.DataFrame({"split": split, "row": index, "proba": fitted.predict_proba(x)})
            parts.append(part.assign(feature_set=source, model=model.name, n_train=len(rows)))
    table = pd.concat(parts, ignore_index=True)
    return table.assign(y_true=data.labels[table["row"].to_numpy()])


def main(args: argparse.Namespace) -> None:
    pokedex = Pokedex.from_json(args.pokedex)
    repair = load_repair(args.repair, pokedex, feature_sets.type_chart())
    data = Dataset.from_jsonl(args.train, pokedex, repair=repair).labelled()
    ((_, splitter),) = FoldPlan(args.n_folds, (args.seed,)).splitters()
    splits = make_splits(splitter, data.labels, data.groups)
    models = build_models(args.models, args.params)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    finished = done(args.out)
    for fold in args.folds:
        train, valid = splits[fold]
        for size in [*args.sizes, *([] if args.no_full else [len(train)])]:
            if (fold, size) in finished:
                continue
            start = time.perf_counter()
            rows = nested_subset(train, data.groups, size)
            table = predictions(data, args.features, models, rows, valid)
            table = table.assign(fold=fold, size=size)
            table.to_csv(args.out, mode="a", header=not args.out.exists(), index=False)
            seconds = time.perf_counter() - start
            print(f"fold {fold} size {size}: {len(rows)} battles, {seconds:.0f} s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="learning-curve predictions of one feature set")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--pokedex", type=Path, default=DEFAULT_POKEDEX)
    parser.add_argument("--repair", default=DEFAULT_REPAIR)
    parser.add_argument("--features", required=True, help="one feature set")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--params", type=Path, nargs="+", help="tuned parameter files (JSON)")
    parser.add_argument("--seed", type=int, default=33, help="seed of the folds")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    sizes_help = "training sizes below the whole fold, which is added unless --no-full"
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES), help=sizes_help)
    parser.add_argument("--no-full", action="store_true", help="for a run split over processes")
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
