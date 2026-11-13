"""The Kaggle notebook of team predictive spikes (1st public, 9th private), run without changing it.

Runs in *their* environment (Python 3.13 and the versions their Kaggle run prints), not in
``.env``:

    PY=<their venv>/bin/python REPO=~/Downloads/predictive-spikes-main
    $PY docs/evaluation/comparison/predictive_spikes.py run --repo $REPO --data D --output O \
        [--ours]
    $PY docs/evaluation/comparison/predictive_spikes.py deploy --repo $REPO --data D --output O
    $PY docs/evaluation/comparison/predictive_spikes.py curve --repo $REPO --data D --output O \
        --subsets <curve_subsets.csv>

Their notebook (``FDS-github.ipynb``, the one that ran on Kaggle) is executed cell by cell in one
namespace, in memory, up to its training cell; the two cells after it only draw the feature
importance and write the binary submission (also into their repository), so they are skipped.
Its text is never changed: the data folder is given through ``POKEMON_DATA_DIR``, which their
loader reads, and the working folder is their repository, where the notebook looks for ``src``.

``run``: the training cell as written (their ``StratifiedKFold``, seed 42, with early stopping
on the held-out fold: ``own_oof.csv``) and the test probabilities of every member and of their
blend, the mean of the five fold models (``test_proba.csv``); with ``--ours`` the same cell
runs again on our seed-33 folds (their ``get_cv_splits`` call replaced by our partition:
``ours_oof.csv``).

``deploy``: the whole notebook once per fold of ours, with the training folds as ``train.jsonl``
and the held-out fold, without labels, as ``test.jsonl`` (``<output>/data_<fold>``): the held-out
battles are predicted as the test set was, by models that never saw them, not even for early
stopping. Writes ``deploy_oof.csv``.

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

NOTEBOOK = "FDS-github.ipynb"
SPLIT = "cv_splits = get_cv_splits(train_battles, y, n_splits=FINAL_FOLDS, seed=RANDOM_SEED, "
TRAINING = "blended_test_probs ="


def code_cells(repo: Path) -> list[str]:
    """The code cells of their notebook up to its training cell, which must be unique."""
    cells = [
        "".join(cell["source"])
        for cell in json.loads((repo / NOTEBOOK).read_text())["cells"]
        if cell["cell_type"] == "code"
    ]
    (last,) = [k for k, cell in enumerate(cells) if TRAINING in cell]
    assert cells[last].count(SPLIT) == 1
    return cells[: last + 1]


def probabilities(ns: dict[str, Any], kind: str) -> pd.DataFrame:
    """The probabilities of every member and of their blend left by the training cell, out of
    fold (``kind="oof"``) or on the test set (``"test"``)."""
    members = ns["oof_probs_map"] if kind == "oof" else ns["test_probs_map"]
    blend = ns["blended_oof_probs"] if kind == "oof" else ns["blended_test_probs"]
    return pd.DataFrame({**members, "blend": blend})


def oof_table(ns: dict[str, Any]) -> pd.DataFrame:
    """The out-of-fold probabilities of the training cell, with the fold of every battle."""
    fold = np.full(len(ns["y"]), -1)
    for k, (_, valid) in enumerate(ns["cv_splits"]):
        fold[valid] = k
    table = pd.DataFrame({"row": range(len(fold)), "fold": fold, "y_true": ns["y"]})
    return pd.concat([table, probabilities(ns, "oof")], axis=1)


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["POKEMON_DATA_DIR"] = str(args.data)
    os.chdir(args.repo)  # their setup cell finds the repository in the working folder
    ns: dict[str, Any] = {"__name__": "__main__"}
    cells = code_cells(args.repo)
    for cell in cells[:-1]:
        exec(cell, ns)
    training = cells[-1]
    if args.ours:
        fold = our_folds()
        ns["__our_folds__"] = [
            (np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in sorted(set(fold))
        ]
        line = training[training.index(SPLIT) :].splitlines()[0]
        exec(training.replace(line, "cv_splits = __our_folds__"), ns)
        oof_table(ns).to_csv(args.output / "ours_oof.csv", index=False)
    exec(training, ns)
    oof_table(ns).to_csv(args.output / "own_oof.csv", index=False)
    test = probabilities(ns, "test")
    test.insert(0, "battle_id", np.asarray(ns["test_ids"]).astype(int))
    test.to_csv(args.output / "test_proba.csv", index=False)
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
        command = [sys.executable, __file__, "run", "--repo", str(args.repo),
                   "--data", str(jobs[name]), "--output", str(output)]  # fmt: skip
        with (output / "log.txt").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"{name} done", flush=True)

    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(one, jobs))


def predictions(args: argparse.Namespace, name: str) -> pd.DataFrame:
    """The test probabilities of the run ``name``, with the row and label of each battle."""
    battles = [json.loads(line) for line in (args.data / "train.jsonl").read_text().splitlines()]
    row = {battle["battle_id"]: r for r, battle in enumerate(battles)}
    table = pd.read_csv(args.output / f"run_{name}" / "test_proba.csv")
    table.insert(0, "row", [row[i] for i in table["battle_id"]])
    table.insert(1, "y_true", [int(battles[r]["player_won"]) for r in table["row"]])
    return table.drop(columns="battle_id")


def deploy(args: argparse.Namespace) -> None:
    fold = our_folds()
    folds = sorted(set(fold))
    jobs = {
        str(k): split_data(args, np.flatnonzero(fold != k), np.flatnonzero(fold == k), str(k))
        for k in folds
    }
    run_notebooks(args, jobs)
    table = pd.concat([predictions(args, str(k)) for k in folds]).sort_values("row")
    table.insert(1, "fold", fold[table["row"].to_numpy()])
    table.to_csv(args.output / "deploy_oof.csv", index=False)
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
        part = predictions(args, str(size))
        parts.append(part.assign(p=part["blend"], size=size, n_train=n_train))
    table = pd.concat(parts)[["size", "n_train", "row", "y_true", "p"]]
    table.to_csv(args.output / "curve.csv", index=False)
    print(f"wrote {args.output / 'curve.csv'}")


def main() -> None:
    """One sub-command per step."""
    root = argparse.ArgumentParser(description="the Kaggle notebook of team predictive spikes")
    commands = root.add_subparsers(dest="command", required=True)
    for name, handler in (("run", run), ("deploy", deploy), ("curve", curve)):
        sub = commands.add_parser(name)
        sub.add_argument("--repo", type=Path, required=True, help="their repository")
        sub.add_argument("--data", type=Path, required=True, help="the four competition files")
        sub.add_argument("--output", type=Path, required=True)
        sub.set_defaults(handler=handler)
    commands.choices["run"].add_argument("--ours", action="store_true")
    for name in ("deploy", "curve"):
        commands.choices[name].add_argument("--jobs", type=int, default=3)
    commands.choices["curve"].add_argument("--subsets", type=Path, required=True)
    args = root.parse_args()
    for name in ("repo", "data", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.handler(args)


if __name__ == "__main__":
    main()
