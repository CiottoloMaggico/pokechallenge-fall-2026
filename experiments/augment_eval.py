"""Does training on truncated copies of training battles help? (prototype)

A random half of a random subsample of the training battles is copied, each
copy cut to its first 20, 22, 24, 26 or 28 turns and keeping the original
label. The copies are used only for fitting:

* every copy shares the group of its original battle, so it always falls
  in the same fold as the original (no copy of a validation battle is ever
  used for training);
* the validation rows of every fold are the untruncated originals only;
* the baseline uses the same folds, trained on the originals alone.

Run from the project root::

    python -m experiments.augment_eval --battles 2000
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import feature_catalog
from data import Dataset, load_records
from domain.pokedex import Pokedex
from experiments.mc_eval import COMPACT_V5
from gym import prepare_folds
from knockout_detector import detector_from_files
from selection import Selector, paired_t

CUTS = (20, 22, 24, 26, 28)


class _Precomputed:
    """A splitter that returns the folds it was given."""

    def __init__(self, folds: Sequence[tuple[np.ndarray, np.ndarray]]) -> None:
        self.folds = list(folds)

    def split(self, x, y, groups):  # noqa: ANN001
        return iter(self.folds)


def _original_folds(
    labels: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    return [
        (np.asarray(tr), np.asarray(va))
        for tr, va in splitter.split(np.zeros(len(labels)), labels, groups)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--train", type=Path, default=Path("input/train.jsonl"))
    parser.add_argument("--pokedex", type=Path, default=Path("assets/species.json"))
    parser.add_argument("--battles", type=int, default=2000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    pokedex = Pokedex.from_json(args.pokedex)
    chart = feature_catalog.type_chart()
    detector = detector_from_files(pokedex, chart, (args.train,))
    records = load_records(args.train)
    rng = np.random.default_rng(args.seed)
    picked = np.sort(rng.choice(len(records), size=args.battles, replace=False))
    originals = [records[i] for i in picked]

    copied = rng.choice(len(originals), size=len(originals) // 2, replace=False)
    copies, source = [], []
    for i in sorted(copied):
        record = originals[i]
        for cut in CUTS:
            if len(record["battle_timeline"]) <= cut:
                continue
            copies.append(
                {
                    **record,
                    "battle_id": f"{record['battle_id']}@t{cut}",
                    "battle_timeline": record["battle_timeline"][:cut],
                }
            )
            source.append(i)

    base = Dataset.from_records(originals, pokedex, repair=detector)
    both = Dataset.from_records([*originals, *copies], pokedex, repair=detector)
    labels = base.labels
    groups = base.duplicate_groups()
    row_groups = np.concatenate([groups, groups[np.asarray(source, dtype=int)]])

    folds = _original_folds(labels, groups, args.folds, args.seed)
    base_folds = [(tr, va) for tr, va in folds]
    aug_folds = []
    for tr, va in folds:
        train_groups = np.unique(groups[tr])
        rows = np.flatnonzero(np.isin(row_groups, train_groups))
        aug_folds.append((rows, va))

    pool = feature_catalog.build("selection_pool_v5")
    base_data = prepare_folds(
        pool,
        base.contexts,
        labels,
        _Precomputed(base_folds),
        groups,
    )
    aug_data = prepare_folds(
        pool,
        both.contexts,
        both.labels,
        _Precomputed(aug_folds),
        row_groups,
    )
    base_acc, base_loss = Selector(base_data, ["logreg"]).score(list(COMPACT_V5))
    aug_acc, aug_loss = Selector(aug_data, ["logreg"]).score(list(COMPACT_V5))
    base_scores = (base_acc[0], base_loss[0])  # the single model's rows
    aug_scores = (aug_acc[0], aug_loss[0])

    d_acc = aug_scores[0] - base_scores[0]
    d_loss = aug_scores[1] - base_scores[1]
    t = paired_t(d_loss)
    print(f"originals {len(originals)}, copies {len(copies)}, folds {len(folds)}")
    print(
        f"accuracy  base {base_scores[0].mean():.4f}  aug {aug_scores[0].mean():.4f}  "
        f"diff {d_acc.mean() * 100:+.2f} points"
    )
    print(
        f"log-loss  base {base_scores[1].mean():.4f}  aug {aug_scores[1].mean():.4f}  "
        f"diff {d_loss.mean():+.5f}  t {t:+.2f}"
    )
    print("per fold accuracy diff:", np.round(d_acc * 100, 2).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
