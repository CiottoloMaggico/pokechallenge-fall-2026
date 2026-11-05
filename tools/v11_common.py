"""Shared steps of ``tools/v11_blend_weight.py`` and ``tools/v11_blend_eval.py``.

* ``fold_tables``: the compact_v5_v11 tables of every fold for each seed
  (StratifiedGroupKFold 5, duplicates grouped, battles repaired as in
  ``main.py``), cached in ``--cache`` (one pickle per seed). This is the
  expensive part: v11 is rebuilt and refitted in every fold.
* ``fold_predictions``: per (seed, fold) the validation labels, the base
  model's probability (fitted on the compact_v5 columns of the training part)
  and v11's probability; cached too, keyed by base model and tuned files.
* ``blend`` / ``scores`` / ``corrected``: logit blend, per-fold accuracy and
  log loss, corrected repeated k-fold t-test (``gym.corrected_ttest``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import feature_catalog  # noqa: E402
import model_catalog  # noqa: E402
from data import Dataset  # noqa: E402
from domain.pokedex import Pokedex  # noqa: E402
from gym import corrected_ttest, prepare_folds  # noqa: E402
from knockout_detector import detector_from_files  # noqa: E402
from tuning import load_params  # noqa: E402

FEATURE_SET = "compact_v5_v11"
COLUMN = "v11_proba"
EPS = 1e-6
RATIO = 0.25  # n_test / n_train with 5 folds


def mode() -> str:
    """Cache tag: how v11's column was computed (V11_FEATURES, V11_FAKE)."""
    if os.environ.get("V11_FAKE") == "1":
        return "fake"
    return os.environ.get("V11_FEATURES", "fold")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train", type=Path, default=ROOT / "input" / "train.jsonl")
    parser.add_argument("--pokedex", type=Path, default=ROOT / "results" / "cache" / "species.json")
    parser.add_argument("--seeds", type=int, nargs="+", help="explicit fold seeds")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42, help="first of --n-seeds consecutive seeds")
    parser.add_argument("--random-seeds", action="store_true")
    parser.add_argument("--jobs", type=int, default=1, help="seeds whose tables are built in parallel")
    parser.add_argument("--base", default="ensemble_from_lr", help="catalog model blended with v11")
    parser.add_argument("--tuned", type=Path, nargs="*", default=[],
                        help="tuned parameter files of the base model / its members")
    parser.add_argument("--cache", type=Path, default=ROOT / "results" / "v11_cache")


def seeds_from(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return list(dict.fromkeys(args.seeds))
    if args.random_seeds:
        return random.SystemRandom().sample(range(1, 2**31 - 1), args.n_seeds)
    return [args.seed + k for k in range(args.n_seeds)]


_SHARED: dict = {}


def _build_seed(seed: int) -> tuple[int, list]:
    ds, out = _SHARED["dataset"], _SHARED["cache"]
    started = time.perf_counter()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = prepare_folds(
        feature_catalog.build(FEATURE_SET), ds.contexts, ds.labels, splitter,
        groups=ds.duplicate_groups(),
        progress=lambda d, t, s: print(f"  seed {seed}: fold {d}/{t} ready ({s:.0f} s)", flush=True),
    )
    with (out / f"tables_{mode()}_s{seed}.pkl").open("wb") as fh:
        pickle.dump(folds, fh)
    print(f"  seed {seed}: tables saved ({time.perf_counter() - started:.0f} s)", flush=True)
    return seed, folds


def fold_tables(args: argparse.Namespace, seeds: list[int]) -> dict[int, list]:
    """{seed: [(x_train, y_train, x_valid, y_valid), ...]} from the cache or built."""
    args.cache.mkdir(parents=True, exist_ok=True)
    tables, todo = {}, []
    for s in seeds:
        path = args.cache / f"tables_{mode()}_s{s}.pkl"
        if path.exists():
            with path.open("rb") as fh:
                tables[s] = pickle.load(fh)
        else:
            todo.append(s)
    if todo:
        print(f"building the fold tables of seeds {todo} (v11 is refitted in every fold)", flush=True)
        pokedex = Pokedex.from_json(args.pokedex)
        detector = detector_from_files(pokedex, feature_catalog.type_chart(), (args.train,))
        _SHARED.update(dataset=Dataset.from_jsonl(args.train, pokedex, repair=detector), cache=args.cache)
        if os.environ.get("V11_FEATURES") == "shared" and os.environ.get("V11_FAKE") != "1":
            from features.v11_bridge import _shared_rows, configure

            configure()
            _shared_rows([], [])  # build v11's shared tables once, before forking
        jobs = args.jobs
        if jobs > 1 and mode() != "fake":
            # LightGBM / XGBoost / CatBoost use OpenMP, which is not fork-safe:
            # forked workers hang at 0% CPU once the parent has used it (the
            # shared v11 tables are built in the parent). Seeds run one after
            # the other; each v11 model already uses FDS_N_JOBS threads.
            print("note: --jobs ignored with v11 (OpenMP is not fork-safe); seeds run in sequence, "
                  "set FDS_N_JOBS to your number of cores", flush=True)
            jobs = 1
        if jobs > 1 and len(todo) > 1 and "fork" in mp.get_all_start_methods():
            with mp.get_context("fork").Pool(min(jobs, len(todo))) as pool:
                built = pool.map(_build_seed, todo, chunksize=1)
        else:
            built = [_build_seed(s) for s in todo]
        tables.update(dict(built))
    return {s: tables[s] for s in seeds}


def _base_model(base: str, tuned: list[Path]):
    params = dict(load_params(p) for p in tuned)
    return model_catalog.build_tuned(base, params)


def fold_predictions(args: argparse.Namespace, seeds: list[int]) -> list[dict]:
    """One dict per (seed, fold): seed, fold, y, p_base, p_v11."""
    tag = hashlib.sha1(json.dumps([args.base, sorted(str(p) for p in args.tuned)]).encode()).hexdigest()[:10]
    tables = None
    out = []
    for s in seeds:
        path = args.cache / f"preds_{mode()}_s{s}_{args.base}_{tag}.pkl"
        if path.exists():
            with path.open("rb") as fh:
                out.extend(pickle.load(fh))
            continue
        tables = tables or fold_tables(args, seeds)
        rows = []
        for k, (xt, yt, xv, yv) in enumerate(tables[s]):
            model = _base_model(args.base, args.tuned).fit(xt.drop(columns=[COLUMN]), yt)
            rows.append({"seed": s, "fold": k, "y": np.asarray(yv),
                         "p_base": model.predict_proba(xv.drop(columns=[COLUMN])),
                         "p_v11": xv[COLUMN].to_numpy(dtype=float)})
        with path.open("wb") as fh:
            pickle.dump(rows, fh)
        out.extend(rows)
    return out


def blend(p_base: np.ndarray, p_v11: np.ndarray, w: float) -> np.ndarray:
    def logit(p):
        p = np.clip(p, EPS, 1 - EPS)
        return np.log(p / (1 - p))
    p_v11 = np.where(np.isnan(p_v11), p_base, p_v11)
    return 1 / (1 + np.exp(-(w * logit(p_base) + (1 - w) * logit(p_v11))))


def scores(rows: list[dict], w: float | dict) -> tuple[np.ndarray, np.ndarray]:
    """Accuracy and log loss per (seed, fold); ``w`` a number or {(seed, fold): w}."""
    acc, ll = [], []
    for r in rows:
        wr = w[(r["seed"], r["fold"])] if isinstance(w, dict) else w
        p = blend(r["p_base"], r["p_v11"], wr)
        acc.append(np.mean((p > 0.5) == r["y"]))
        ll.append(log_loss(r["y"], np.clip(p, EPS, 1 - EPS), labels=[0, 1]))
    return np.array(acc), np.array(ll)


def per_seed(rows: list[dict], values: np.ndarray) -> np.ndarray:
    seeds = np.array([r["seed"] for r in rows])
    return np.array([values[seeds == s].mean() for s in dict.fromkeys(seeds)])


def corrected(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Corrected t and p of b - a over the (seed, fold) pairs."""
    return corrected_ttest(b - a, RATIO)
