"""The nested training subsets of our learning curve, for the second team's models.

    python docs/evaluation/comparison/curve_subsets.py --out results/comparison/curve_subsets.csv

Writes ``size,row``: for every size, the training battles of fold 0 (seed 33) that
``learning_curve.py`` fits on (``nested_subset``, whole duplicate groups), so that
``second_place.py curve`` trains their models on exactly the same battles. The duplicate groups
do not depend on the repair, so none is applied; the folds are checked against ``oof.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from learning_curve import nested_subset  # noqa: E402

from data import Dataset  # noqa: E402
from domain.pokedex import Pokedex  # noqa: E402
from folds import FoldPlan, make_splits  # noqa: E402
from settings import DEFAULT_POKEDEX, DEFAULT_TRAIN  # noqa: E402

SIZES = (500, 1000, 2000, 3000, 4000, 5000, 6000, 7000)
FOLD, SEED = 0, 33


def main(out: Path) -> None:
    data = Dataset.from_jsonl(DEFAULT_TRAIN, Pokedex.from_json(DEFAULT_POKEDEX)).labelled()
    ((_, splitter),) = FoldPlan(5, (SEED,)).splitters()
    train, valid = make_splits(splitter, data.labels, data.groups)[FOLD]
    oof = pd.read_csv(HERE.parent / "data" / "oof.csv")
    if sorted(valid.tolist()) != sorted(oof.loc[oof["fold"] == FOLD, "row"].tolist()):
        raise SystemExit("the folds differ from those of docs/evaluation/data/oof.csv")
    parts = [pd.DataFrame({"size": size, "row": nested_subset(train, data.groups, size)})
             for size in SIZES]  # fmt: skip
    parts.append(pd.DataFrame({"size": len(train), "row": np.sort(train)}))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts).to_csv(out, index=False)
    print(pd.concat(parts).groupby("size").size().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nested subsets of our learning curve")
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
