"""The notebooks of team Fantastic Four (3rd public, 8th private), run without changing them.

Runs in *their* environment (Python 3.13 and the versions their notebooks print), not in ``.env``:

    PY=<their venv>/bin/python REPO=~/Downloads/FDS-Challenge-2026
    $PY docs/evaluation/comparison/fantastic_four.py run --notebook "$REPO/submission 8 - ..." \
        --data D --output O [--cv own ours]
    $PY docs/evaluation/comparison/fantastic_four.py deploy --notebook ... --data D --output O
    $PY docs/evaluation/comparison/fantastic_four.py curve --notebook ... --data D --output O \
        --subsets <curve_subsets.csv>

A notebook is executed cell by cell in one namespace, in memory: the only change to its text is
the path of the competition data. ``run``: the definitions at the top of its validation cell
always run; with ``--cv own`` the whole cell runs as written (their ``StratifiedKFold``, seed 42:
``own_oof.csv``); with ``--cv ours`` the same cell runs on our seed-33 folds
(``cv.split(X_train, y)`` replaced by our partition: ``ours_oof.csv``); then its test cell writes
``submission.csv`` and ``test_proba.csv``.

``deploy``: the whole notebook once per fold of ours, with the training folds as ``train.jsonl``
and the held-out fold, without labels, as ``test.jsonl`` (``<output>/data_<fold>``). Every table
is then built as at submission time, from the training battles only, whereas their validation
cell reuses tables built on all 10,000 battles. Writes ``deploy_oof.csv``.

``curve``: the same with the nested subsets of our learning curve (``curve_subsets.py``) as
train and our fold 0 as test. Writes ``curve.csv``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from second_place import our_folds  # noqa: E402

KAGGLE = "Path('/kaggle/input/competitions/fds-pokemon-battles-prediction-2026')"
CV_LINE = "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)"
CV_SPLIT = "cv.split(X_train, y)"


def code_cells(notebook: Path, data: Path) -> list[str]:
    """The code cells of a notebook, with the data folder in place of Kaggle's."""
    cells = [
        "".join(cell["source"])
        for cell in json.loads(notebook.read_text())["cells"]
        if cell["cell_type"] == "code"
    ]
    assert sum(KAGGLE in cell for cell in cells) == 1
    return [cell.replace(KAGGLE, f"Path({str(data)!r})") for cell in cells]


def oof_table(ns: dict[str, Any], folds: list[tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """The out-of-fold probabilities left in ``oof`` by their validation cell."""
    fold = np.full(len(ns["y"]), -1)
    for k, (_, valid) in enumerate(folds):
        fold[valid] = k
    return pd.DataFrame({"row": range(len(fold)), "fold": fold, "y_true": ns["y"], "p": ns["oof"]})


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    ns: dict[str, Any] = {"__name__": "__main__"}
    for cell in code_cells(args.notebook, args.data):
        if CV_LINE in cell:
            assert cell.count(CV_SPLIT) == 1
            exec(cell[: cell.index(CV_LINE)], ns)  # definitions the test cell needs too
            if "own" in args.cv:
                exec(cell, ns)
                own = list(ns["StratifiedKFold"](5, shuffle=True, random_state=42).split(
                    ns["X_train"], ns["y"]))  # fmt: skip
                oof_table(ns, own).to_csv(args.output / "own_oof.csv", index=False)
            if "ours" in args.cv:
                fold = our_folds()
                ns["__our_folds__"] = [
                    (np.flatnonzero(fold != k), np.flatnonzero(fold == k))
                    for k in sorted(set(fold))
                ]
                exec(cell.replace(CV_SPLIT, "__our_folds__"), ns)
                table = oof_table(ns, ns["__our_folds__"])
                table.to_csv(args.output / "ours_oof.csv", index=False)
            continue
        if "test.jsonl" in cell:
            os.chdir(args.output)  # their test cell writes submission.csv in the working folder
        exec(cell, ns)
    ids = ns["test_df"]["battle_id"].astype(int).to_numpy()
    pd.DataFrame({"battle_id": ids, "p": ns["p_test"]}).to_csv(
        args.output / "test_proba.csv", index=False
    )
    print(f"{args.output}: done", flush=True)


def split_data(args: argparse.Namespace, train: np.ndarray, test: np.ndarray, name: str) -> Path:
    """Their four files with the rows ``train`` as train and ``test`` (no labels) as test."""
    folder = args.output / f"data_{name}"
    folder.mkdir(parents=True, exist_ok=True)
    lines = (args.data / "train.jsonl").read_text().splitlines(keepends=True)
    (folder / "train.jsonl").write_text("".join(lines[row] for row in train))
    held = [json.loads(lines[row]) for row in test]
    with (folder / "test.jsonl").open("w") as out:
        for battle in held:
            battle.pop("player_won")
            out.write(json.dumps(battle) + "\n")
    ids = [battle["battle_id"] for battle in held]
    pd.DataFrame({"battle_id": ids, "player_won": 0}).to_csv(
        folder / "sample_submission.csv", index=False
    )
    chart = folder / "type_chart_gen2.csv"
    if not chart.exists():
        chart.symlink_to((args.data / "type_chart_gen2.csv").resolve())
    return folder


def run_notebooks(args: argparse.Namespace, jobs: dict[str, Path]) -> None:
    """The notebook once per data folder of ``jobs`` (name: folder), ``args.jobs`` at a time."""

    def one(name: str) -> None:
        output = args.output / f"run_{name}"
        output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, __file__, "run", "--notebook", str(args.notebook),
                   "--data", str(jobs[name]), "--output", str(output)]  # fmt: skip
        with (output / "log.txt").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"{name} done", flush=True)

    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(one, jobs))


def predictions(args: argparse.Namespace, name: str) -> pd.DataFrame:
    """The probabilities of the run ``name``, with the row and label of each battle."""
    battles = [json.loads(line) for line in (args.data / "train.jsonl").read_text().splitlines()]
    row = {battle["battle_id"]: r for r, battle in enumerate(battles)}
    table = pd.read_csv(args.output / f"run_{name}" / "test_proba.csv")
    table.insert(0, "row", [row[i] for i in table["battle_id"]])
    table["y_true"] = [int(battles[r]["player_won"]) for r in table["row"]]
    return table


def deploy(args: argparse.Namespace) -> None:
    fold = our_folds()
    folds = sorted(set(fold))
    jobs = {
        str(k): split_data(args, np.flatnonzero(fold != k), np.flatnonzero(fold == k), str(k))
        for k in folds
    }
    run_notebooks(args, jobs)
    table = pd.concat([predictions(args, str(k)).assign(fold=k) for k in folds]).sort_values("row")
    assert (table["fold"].to_numpy() == fold[table["row"].to_numpy()]).all()
    table[["row", "fold", "y_true", "p"]].to_csv(args.output / "deploy_oof.csv", index=False)
    print(f"wrote {args.output / 'deploy_oof.csv'}")


def curve(args: argparse.Namespace) -> None:
    subsets = pd.read_csv(args.subsets)
    valid = np.flatnonzero(our_folds() == 0)
    sizes = sorted(set(subsets["size"]))
    jobs = {}
    for size in sizes:
        train = np.sort(subsets.loc[subsets["size"] == size, "row"].to_numpy())
        jobs[str(size)] = split_data(args, train, valid, str(size))
    run_notebooks(args, jobs)
    parts = []
    for size in sizes:
        n_train = int((subsets["size"] == size).sum())
        parts.append(predictions(args, str(size)).assign(size=size, n_train=n_train))
    table = pd.concat(parts)[["size", "n_train", "row", "y_true", "p"]]
    table.to_csv(args.output / "curve.csv", index=False)
    print(f"wrote {args.output / 'curve.csv'}")


def main() -> None:
    """One sub-command per step."""
    root = argparse.ArgumentParser(description="the notebooks of team Fantastic Four")
    commands = root.add_subparsers(dest="command", required=True)
    for name, handler in (("run", run), ("deploy", deploy), ("curve", curve)):
        sub = commands.add_parser(name)
        sub.add_argument("--notebook", type=Path, required=True)
        sub.add_argument("--data", type=Path, required=True, help="the four competition files")
        sub.add_argument("--output", type=Path, required=True)
        sub.set_defaults(handler=handler)
    commands.choices["run"].add_argument("--cv", nargs="*", default=[], choices=["own", "ours"])
    for name in ("deploy", "curve"):
        commands.choices[name].add_argument("--jobs", type=int, default=3)
    commands.choices["curve"].add_argument("--subsets", type=Path, required=True)
    args = root.parse_args()
    for name in ("notebook", "data", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.handler(args)


if __name__ == "__main__":
    main()
