"""The second-placed team's models on our folds, run from their repository without changing it.

Runs in *their* environment (Python 3.12 and their ``requirements.txt``), not in ``.env``:

    PY=<their venv>/bin/python REPO=~/Downloads/FDS-Pokemon-Challenge-2026-2027
    $PY docs/evaluation/comparison/second_place.py cv --repo $REPO --data D --output O
    $PY docs/evaluation/comparison/second_place.py stage --repo $REPO --data D --output O
    $PY docs/evaluation/comparison/second_place.py test --repo $REPO --data D --output P

``cv``: their ``prepare`` (mode ``cv``, so no test battle is read), then their partitions are
replaced by ours (seed 33, the folds of ``docs/evaluation/data/oof.csv``) under the key ``33``
and their own worker (``run.py cv --worker 33 <fold>``) fits the five views of each fold. Their
groups are identical timelines, ours whole battles: three timelines (six battles that differ
only in P1's team details) are split by our folds, so those battles get one group per fold, or
their leakage assertion stops the worker. Writes ``components.csv`` (the five views,
out-of-fold). ``--feature-fixes`` as in ``run.py`` (their two final submissions use it).

``stage``: their second stage on the same folds: the similar-team rates of
``similar_team_features`` (their inner folds of each outer training fold), Spikes at turn 30 and
the logistic regression on logit(tree probability), fitted on the training folds and applied to
the held-out fold. Writes ``second_oof.csv``: per battle, ``five_view``, ``tactical``,
``team_similarity`` and ``tactical_team_spikes``.

``curve``: their five views trained on the nested subsets of our learning curve
(``curve_subsets.py``: fold 0, the same battles as ours) and evaluated on fold 0. Their worker
wants every battle of its file in the fold, so each size gets its own ``train.jsonl`` (the subset
and fold 0, in the original order) under ``<output>/data_<size>``. Writes ``curve.csv``.

``test``: the test probabilities of a finished ``run.py predict --feature-fixes`` (their own
protocol: three partitions, routed test battles), with the second stage fitted exactly as in
``second_stage.run``; checks that the labels equal ``results/submitted`` and writes
``second_test.csv``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEED = 33  # our fold assignment, stored under this key next to their three partitions
OOF = Path(__file__).resolve().parents[1] / "data" / "oof.csv"
TACTICAL = 2  # index of the tactical view in their COMPONENTS
SETTINGS = {"PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}


def their(repo: Path, module: str) -> Any:
    """A module of their ``src`` package (imported, never modified)."""
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return importlib.import_module(f"src.{module}")


def our_folds() -> np.ndarray:
    """Our fold (seed 33) of every training battle, in the order of ``train.jsonl``."""
    oof: Any = pd.read_csv(OOF)
    oof = oof[oof["seed"] == SEED].sort_values("row")
    assert oof["row"].tolist() == list(range(len(oof)))
    return oof["fold"].to_numpy()


def split_groups(groups: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Their groups, with one group per fold for the timelines that our folds split."""
    groups = groups.astype(object)
    frame = pd.DataFrame({"group": groups, "fold": fold})
    split = frame.groupby("group")["fold"].transform("nunique").to_numpy() > 1
    groups[split] = [f"{g}|fold{f}" for g, f in zip(groups[split], fold[split], strict=True)]
    return np.asarray(groups, dtype=str)


Partition = list[tuple[np.ndarray, np.ndarray]]


def prepare(repo: Path, data: Path, output: Path, fold: np.ndarray, partition: Partition) -> None:
    """Their ``prepare`` (mode cv), then ``partition`` stored under the key ``SEED``.

    ``fold``: our fold of every battle of ``data`` (the groups are split by it, the same way
    in every run, so that a run differs from another only by its partition)."""
    pipeline = their(repo, "pipeline")
    pipeline.prepare(data, output, "cv")
    path = output / "inputs.pkl.gz"
    inputs = pipeline.read_pickle(path)
    inputs["groups"] = split_groups(inputs["groups"], fold)
    inputs["partitions"][SEED] = partition
    assigned = np.full(len(fold), -1)
    for k, (_, valid) in enumerate(partition):
        assigned[valid] = k
    inputs["assignments"][SEED] = assigned
    inputs["routing"][SEED] = np.zeros(0, dtype=int)  # mode cv: no test battle
    pipeline.save_pickle(path, inputs)


def run_worker(args: argparse.Namespace, data: Path, output: Path, fold: int) -> None:
    """Their own worker (``run.py cv --worker``) on one fold of the partition ``SEED``."""
    fixes = ["--feature-fixes"] if args.feature_fixes else []
    command = [sys.executable, str(args.repo / "run.py"), "cv", "--data", str(data),
               "--output", str(output), "--worker", str(SEED), str(fold), *fixes]  # fmt: skip
    with (output / f"{SEED}_{fold}.log").open("a") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"{output.name} fold {fold} done", flush=True)


def cv(args: argparse.Namespace) -> None:
    pipeline = their(args.repo, "pipeline")
    fold = our_folds()
    partition = [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in sorted(set(fold))]
    prepare(args.repo, args.data, args.output, fold, partition)
    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(lambda k: run_worker(args, args.data, args.output, k), range(len(partition))))
    inputs = pipeline.read_pickle(args.output / "inputs.pkl.gz")
    names = their(args.repo, "validation").COMPONENTS
    table = pd.DataFrame(np.nan, index=range(len(inputs["y"])), columns=names)
    fold_of = np.full(len(table), -1)
    for k, (train, valid) in enumerate(inputs["partitions"][SEED]):
        with np.load(args.output / f"{SEED}_{k}.npz") as result:
            assert np.array_equal(result["tr"], train) and np.array_equal(result["va"], valid)
            table.iloc[valid] = result["cv_components"]
            fold_of[valid] = k
    assert (fold_of >= 0).all() and np.isfinite(table.to_numpy()).all()
    table.insert(0, "y_true", inputs["y"])
    table.insert(0, "fold", fold_of)
    table.insert(0, "row", range(len(table)))
    table.to_csv(args.output / "components.csv", index=False, float_format="%.8f")
    print(f"wrote {args.output / 'components.csv'}")


def curve_data(args: argparse.Namespace, size: int, rows: np.ndarray) -> Path:
    """Their four files with a ``train.jsonl`` of the given rows only."""
    folder = args.output / f"data_{size}"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("test.jsonl", "type_chart_gen2.csv", "sample_submission.csv"):
        if not (folder / name).exists():
            (folder / name).symlink_to((args.data / name).resolve())
    keep = set(rows.tolist())
    with (args.data / "train.jsonl").open() as source, (folder / "train.jsonl").open("w") as out:
        out.writelines(line for k, line in enumerate(source) if k in keep)
    return folder


def curve(args: argparse.Namespace) -> None:
    names = their(args.repo, "validation").COMPONENTS
    subsets = pd.read_csv(args.subsets)
    fold = our_folds()
    valid = np.flatnonzero(fold == 0)
    jobs = []
    for size in sorted(set(subsets["size"])):
        rows = np.union1d(
            subsets.loc[subsets["size"] == size, "row"].to_numpy(), valid
        )  # original order of train.jsonl
        local = np.where(np.isin(rows, valid), 0, 1)  # fold 0 is the validation fold
        partition = [(np.flatnonzero(local == 1), np.flatnonzero(local == 0))]
        data, output = curve_data(args, size, rows), args.output / f"run_{size}"
        prepare(args.repo, data, output, fold[rows], partition)
        jobs.append((size, rows, partition[0][1], data, output))
    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(lambda job: run_worker(args, job[3], job[4], 0), jobs))
    parts = []
    for size, rows, local_valid, _, output in jobs:
        with np.load(output / f"{SEED}_0.npz") as result:
            part = pd.DataFrame(result["cv_components"], columns=names)
            n_train = len(result["tr"])
        part.insert(0, "y_true", their(args.repo, "pipeline").read_pickle(
            output / "inputs.pkl.gz")["y"][local_valid])  # fmt: skip
        part.insert(0, "row", rows[local_valid])
        parts.append(part.assign(size=size, n_train=n_train))
    pd.concat(parts).to_csv(args.output / "curve.csv", index=False, float_format="%.8f")
    print(f"wrote {args.output / 'curve.csv'}")


def stage_columns(stage: Any, base: np.ndarray, rates: np.ndarray, spikes: np.ndarray | None):
    """The columns of their logistic regression: logit(base), the two rates, optionally Spikes."""
    columns = [stage.logit(base), rates]
    return np.column_stack(columns if spikes is None else [*columns, spikes])


def stage(args: argparse.Namespace) -> None:
    from sklearn.linear_model import LogisticRegression

    pipeline, stage_module = their(args.repo, "pipeline"), their(args.repo, "second_stage")
    observations = their(args.repo, "observations")
    inputs = pipeline.read_pickle(args.output / "inputs.pkl.gz")
    components = pd.read_csv(args.output / "components.csv")
    y, groups = inputs["y"], inputs["groups"]
    train_battles = observations.load(args.data / "train.jsonl")
    sim = stage_module.SimilarTeams(train_battles, [])
    sim.y = y
    rates = np.zeros((len(y), 2), np.float32)
    for train, valid in inputs["partitions"][SEED]:
        fits = stage_module.inner_folds(y, groups, train)
        rates[valid] = sum(sim.rates(f, valid) for f in fits) / len(fits)
    spikes = stage_module.spikes_feature(train_battles)
    views = components.iloc[:, 3:].to_numpy()
    out = components[["row", "fold", "y_true"]].copy()
    out["five_view"], out["tactical"] = views.mean(axis=1), views[:, TACTICAL]
    for name, base, extra in (("team_similarity", out["five_view"], None),
                              ("tactical_team_spikes", out["tactical"], spikes)):  # fmt: skip
        x = stage_columns(stage_module, np.asarray(base), rates, extra)
        p = np.zeros(len(y))
        for train, valid in inputs["partitions"][SEED]:
            model = LogisticRegression(C=1.0, max_iter=2000).fit(x[train], y[train])
            p[valid] = model.predict_proba(x[valid])[:, 1]
        out[name] = p
    out.to_csv(args.output / "second_oof.csv", index=False, float_format="%.8f")
    print(f"wrote {args.output / 'second_oof.csv'}")


def own_report(x: list[np.ndarray], y: np.ndarray, groups: np.ndarray) -> list[np.ndarray]:
    """The out-of-fold rows of the report of ``second_stage.run``, one array per partition."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold

    out = []
    for part in x:
        p = np.zeros(len(y))
        split = StratifiedGroupKFold(5, shuffle=True, random_state=7)
        for a, b in split.split(part, y, groups):
            model = LogisticRegression(C=1.0, max_iter=2000).fit(part[a], y[a])
            p[b] = model.predict_proba(part[b])[:, 1]
        out.append(p)
    return out


def test(args: argparse.Namespace) -> None:
    """Their ``second_stage.run`` for both final submissions, keeping the probabilities, and
    the out-of-fold rows of its report (their own protocol: ``second_own_oof.csv``)."""
    from sklearn.linear_model import LogisticRegression

    pipeline, stage_module = their(args.repo, "pipeline"), their(args.repo, "second_stage")
    observations = their(args.repo, "observations")
    inputs = pipeline.read_pickle(args.output / "inputs.pkl.gz")
    oof, test_p = np.load(args.output / "oof.npz"), np.load(args.output / "test_probabilities.npz")
    train_battles = observations.load(args.data / "train.jsonl")
    test_battles = observations.load(args.data / "test.jsonl")
    y = inputs["y"]
    sim = stage_module.SimilarTeams(train_battles, test_battles)
    rows = np.arange(len(y), len(y) + len(test_battles))
    team_oof, team_test = stage_module.similar_team_features(sim, inputs, rows)
    spikes_train = stage_module.spikes_feature(train_battles)
    spikes_test = stage_module.spikes_feature(test_battles)
    out = pd.DataFrame({"battle_id": test_p["battle_id"], "five_view": test_p["p"]})
    tactical_test = test_p["components"][:, :, TACTICAL].mean(0)
    out["tactical"] = tactical_test
    settings = {
        "team_similarity": ([c.mean(1) for c in oof["components"]], test_p["p"], None),
        "tactical_team_spikes": ([c[:, TACTICAL] for c in oof["components"]], tactical_test,
                                 (spikes_train, spikes_test)),
    }  # fmt: skip
    submitted = args.repo / "results" / "submitted"
    own = pd.DataFrame({"row": range(len(y)), "y_true": y})
    seeds = their(args.repo, "validation").SEEDS
    for name, (base_oof, base_test, spikes) in settings.items():
        train_x = [
            stage_columns(stage_module, b, t, None if spikes is None else spikes[0])
            for b, t in zip(base_oof, team_oof, strict=True)
        ]
        test_x = stage_columns(stage_module, base_test, team_test,
                               None if spikes is None else spikes[1])  # fmt: skip
        model = LogisticRegression(C=1.0, max_iter=2000)
        model.fit(np.vstack(train_x), np.concatenate([y] * len(train_x)))
        out[name] = model.predict_proba(test_x)[:, 1]
        for seed, p in zip(seeds, own_report(train_x, y, inputs["groups"]), strict=True):
            own[f"{name}_{seed}"] = p
        labels = pd.read_csv(submitted / f"submission_{name}.csv")
        mine = pd.Series((out[name] >= 0.5).astype(int).to_numpy(), index=out["battle_id"])
        same = (mine.loc[labels["battle_id"]].to_numpy() == labels["player_won"]).sum()
        print(json.dumps({"submission": name, "same_labels": int(same), "of": len(labels)}))
    own.to_csv(args.output / "second_own_oof.csv", index=False, float_format="%.8f")
    out.to_csv(args.output / "second_test.csv", index=False, float_format="%.8f")
    print(f"wrote {args.output / 'second_test.csv'}")


def parser() -> argparse.ArgumentParser:
    """One sub-command per step."""
    root = argparse.ArgumentParser(description="the second-placed team's models on our folds")
    commands = root.add_subparsers(dest="command", required=True)
    for name, handler in (("cv", cv), ("curve", curve), ("stage", stage), ("test", test)):
        sub = commands.add_parser(name)
        sub.add_argument("--repo", type=Path, required=True, help="their repository")
        sub.add_argument("--data", type=Path, required=True, help="their four competition files")
        sub.add_argument("--output", type=Path, required=True)
        sub.set_defaults(handler=handler)
        if name in ("cv", "curve"):
            sub.add_argument("--jobs", type=int, default=2)
            sub.add_argument("--feature-fixes", action="store_true")
        if name == "curve":
            sub.add_argument("--subsets", type=Path, required=True, help="curve_subsets.py")
    return root


def main() -> None:
    """Re-run with their process settings (as ``run.py``), then run one step."""
    fixes = "1" if "--feature-fixes" in sys.argv else "0"
    settings = {**SETTINGS, "FDS_FEATURE_FIXES": fixes}
    if any(os.environ.get(name) != value for name, value in settings.items()):
        os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, **settings})
    args = parser().parse_args()
    for name in ("repo", "data", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.handler(args)


if __name__ == "__main__":
    main()
