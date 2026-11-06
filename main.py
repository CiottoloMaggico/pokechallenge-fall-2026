"""Command line entry point: wires data, feature sets, models and the gym.

Commands::

    python main.py check    --train input/train.jsonl --test input/test.jsonl
    python main.py evaluate --train input/train.jsonl --features hp30 \\
        --models logreg hgb ensemble --test input/test.jsonl --out results/hp30
    python main.py train    --train input/train.jsonl --features hp30 \\
        --models ensemble --test input/test.jsonl --out runs
    python main.py predict  --run runs/hp30__ensemble --test input/test.jsonl \\
        --out submission.csv
    python main.py evaluate --train input/train.jsonl --features core \\
        --models logreg --turns 5 10 15 20 25 30 --out results/by_turn
    python main.py tune     --features compact --model hgb --trials 200
    python main.py evaluate --features compact --models hgb \\
        --params runs/tuning/compact__hgb.json

``--test`` in ``evaluate`` and ``train`` is optional: its battles (never
its labels, which it does not have) feed label-free stateful features.
Duplicated battles are always kept in the same fold. Every battle is
repaired for the P2 knockouts the records omit (``domain.transitions``).

``tune`` searches a model's hyperparameters with Optuna (``pip install
optuna``) on folds seeded differently from ``evaluate``, stores the study
in SQLite (several terminals can share it) and writes the best
parameters to ``runs/tuning/<set>__<model>.json``; ``--params`` (one or
more files) then builds those models with them, also inside ensembles
(named ``<model>-tuned``).

``evaluate --n-seeds 5 --jobs 2`` repeats the cross-validation on five
fold assignments (seeds 42..46; ``--seeds 1 7 99`` picks them,
``--random-seeds`` draws them) and adds ``accuracy_seed_std`` and
``log_loss_seed_std``, how much the result moves with the luck of the
folds, plus an accuracy-by-seed table; the seeds are in ``meta.json`` and
in the ``seed`` column of ``folds.csv`` and ``predictions.csv``.

``--turns`` samples every feature set at each given turn (sets named
``core@t5``, ``core@t10``, ...); ``evaluate`` then also prints and
exports ``accuracy_by_turn.csv``, ready for an accuracy-vs-turn plot.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import random
import sys
import time
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from artifacts import TrainedRun, save_runs
from data import Dataset
from domain.battle import BattleContext
from domain.pokedex import Pokedex, species_without_stats
from domain.repair import TimelineRepair
from feature_set import FeatureSet
from features import sets as feature_sets
from gym import FoldData, Gym, GymResult, prepare_folds, seed_table
from interfaces import Model
from knockout_detector import detector_from_files
from models import ENSEMBLES, MODELS, build_tuned, search_spaces
from tuning import Tuner, cached_folds, load_params, save_best

DEFAULT_POKEDEX = Path("results/cache/species.json")
DEFAULT_TRAIN = Path("input/train.jsonl")
DEFAULT_TEST = Path("input/test.jsonl")
DEFAULT_FOLDS = 5
DEFAULT_SEED = 42
TUNE_SEED = 2027  # tuning folds differ from evaluation folds
STUDY_DB = "optuna.db"
SUBMISSION_COLUMNS = ("battle_id", "player_won")
MAX_SEED_COLUMNS = 10  # more seeds: the accuracy-by-seed table is only exported

logger = logging.getLogger("main")


# ----------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------
def check(args: argparse.Namespace) -> int:
    """Load the data and report what the pipeline will do with it."""
    pokedex = Pokedex.from_json(args.pokedex)
    train = Dataset.from_jsonl(args.train, pokedex, repair=_repair(args))
    groups = train.duplicate_groups()
    print(f"pokedex: {len(pokedex)} species (source sha256 {pokedex.source_sha256})")
    print(
        f"train: {len(train)} battles, P1 win rate {train.labels.mean():.3f}, "
        f"{len(train) - len(set(groups))} duplicates, "
        f"{_without_stats(train, pokedex)}"
    )
    if args.test:
        test = Dataset.from_jsonl(args.test, pokedex, repair=_repair(args))
        print(
            f"test: {len(test)} battles, labelled {test.is_labelled}, "
            f"{_without_stats(test, pokedex)}"
        )
    return 0


def _without_stats(dataset: Dataset, pokedex: Pokedex) -> str:
    missing = species_without_stats((c.record for c in dataset), pokedex)
    return f"species without stats: {missing or 'none'}"


def evaluate(args: argparse.Namespace) -> int:
    """Cross-validate every feature set with every model and export.

    With several seeds (``--n-seeds``, ``--seeds`` or ``--random-seeds``)
    the whole cross-validation is repeated on fold assignments drawn with
    each seed (``--jobs`` of them in parallel); the table then reports,
    besides the fold std, the std across seeds of accuracy and log loss.
    """
    pokedex = Pokedex.from_json(args.pokedex)
    train = Dataset.from_jsonl(args.train, pokedex, repair=_repair(args))
    unlabeled = _unlabeled(args, pokedex)
    seeds = _seeds(args)
    print(f"seeds: {seeds}", flush=True)
    feature_sets = _feature_sets(args.features, args.turns)
    models = _models(args.models, args.params)
    data = (train.contexts, train.labels, train.ids, train.duplicate_groups(), unlabeled)
    if args.jobs > 1 and len(seeds) > 1 and "fork" in mp.get_all_start_methods():
        # forked workers share the replayed battles (copy on write) instead
        # of receiving a pickled copy each, which would multiply the memory
        _SHARED["job"] = (models, feature_sets, args.folds, data)
        with mp.get_context("fork").Pool(min(args.jobs, len(seeds))) as pool:
            runs = pool.map(_evaluate_shared, seeds, chunksize=1)
    else:
        runs = [_evaluate_seed(models, feature_sets, args.folds, s, data) for s in seeds]
    result = runs[0] if len(seeds) == 1 else GymResult.combine(dict(zip(seeds, runs, strict=True)))
    summary = result.summary().sort_values("accuracy_mean", ascending=False)
    print(summary.to_string(index=False, float_format="{:.4f}".format))
    by_seed = None
    if len(seeds) > 1:
        by_seed = seed_table(result).set_index(["feature_set", "model"]).loc[
            list(zip(summary["feature_set"], summary["model"], strict=True))
        ]
        if len(seeds) <= MAX_SEED_COLUMNS:
            print("\naccuracy by seed (mean over the folds):")
            print(by_seed.to_string(float_format="{:.4f}".format))
    paired = None
    if args.reference:
        reference = tuple(args.reference.split("/", 1))
        paired = pd.concat(
            [
                result.paired(reference, metric).assign(metric=metric)
                for metric in ("accuracy", "log_loss")
            ],
            ignore_index=True,
        )
        paired = paired[
            ~((paired["feature_set"] == reference[0]) & (paired["model"] == reference[1]))
        ]
        print(f"\npaired vs {args.reference} (seed x fold, corrected t-test):")
        print(paired.to_string(index=False, float_format="{:.4f}".format))
    by_turn = _accuracy_by_turn(summary) if args.turns else None
    if by_turn is not None:
        print(by_turn.to_string(float_format="{:.4f}".format))
    if args.out:
        result.export(args.out)
        if paired is not None:
            paired.to_csv(args.out / "paired.csv", index=False)
        summary.to_csv(args.out / "summary.csv", index=False)
        if by_seed is not None:
            by_seed.to_csv(args.out / "accuracy_by_seed.csv")
        if by_turn is not None:
            by_turn.to_csv(args.out / "accuracy_by_turn.csv")
        print(f"results exported to {args.out}")
    return 0


def _evaluate_seed(
    models: Sequence[Model],
    feature_sets: Sequence[FeatureSet[BattleContext]],
    n_folds: int,
    seed: int,
    data: tuple,
) -> GymResult:
    """One full cross-validation with folds drawn with ``seed``."""
    contexts, labels, ids, groups, unlabeled = data
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    gym = Gym(models, feature_sets, splitter)
    started = time.perf_counter()
    result = gym.evaluate(contexts, labels, ids=ids, groups=groups, unlabeled=unlabeled)
    print(f"  seed {seed} done ({time.perf_counter() - started:.0f} s)", flush=True)
    return result


_SHARED: dict[str, tuple] = {}


def _evaluate_shared(seed: int) -> GymResult:
    models, feature_sets, n_folds, data = _SHARED["job"]
    return _evaluate_seed(models, feature_sets, n_folds, seed, data)


def _seeds(args: argparse.Namespace) -> list[int]:
    """The seeds of the fold assignments: ``--seeds`` as given, else
    ``--n-seeds`` random ones (``--random-seeds``) or consecutive ones
    starting from ``--seed`` (default: just ``--seed``)."""
    if args.seeds:
        return list(dict.fromkeys(args.seeds))
    if args.n_seeds < 1:
        raise SystemExit("--n-seeds must be at least 1")
    if args.random_seeds:
        rng = random.SystemRandom()
        return rng.sample(range(1, 2**31 - 1), args.n_seeds)
    return [args.seed + k for k in range(args.n_seeds)]


def train(args: argparse.Namespace) -> int:
    """Fit every combination on all training data and save the runs."""
    pokedex = Pokedex.from_json(args.pokedex)
    repair = _repair(args)
    dataset = Dataset.from_jsonl(args.train, pokedex, repair=repair)
    gym = Gym(
        _models(args.models, args.params), _feature_sets(args.features, args.turns)
    )
    runs = gym.train(
        dataset.contexts,
        dataset.labels,
        unlabeled=_unlabeled(args, pokedex),
        groups=dataset.duplicate_groups(),
        repair=repair.describe(),
    )
    for path in save_runs(runs, args.out):
        print(f"saved {path}")
    return 0


def tune(args: argparse.Namespace) -> int:
    """Search the hyperparameters of one model on one feature set."""
    features = feature_sets.build(args.features)
    unlabeled_tag = "_unl" if args.test else ""
    cache = (
        args.storage_dir
        / "folds"
        / (f"{features.name}__k{args.folds}_s{args.seed}{unlabeled_tag}.pkl")
    )

    def prepare() -> list[FoldData]:
        print(f"preparing the features of {args.folds} folds (once, then cached)")
        pokedex = Pokedex.from_json(args.pokedex)
        train_set = Dataset.from_jsonl(args.train, pokedex, repair=_repair(args))
        splitter = StratifiedGroupKFold(
            n_splits=args.folds, shuffle=True, random_state=args.seed
        )
        return prepare_folds(
            features,
            train_set.contexts,
            train_set.labels,
            splitter,
            groups=train_set.duplicate_groups(),
            unlabeled=_unlabeled(args, pokedex),
            progress=_report_fold,
        )

    folds = cached_folds(cache, prepare, refresh=args.refresh)
    print(f"fold features ready ({cache}); starting the trials")
    tuner = Tuner(args.model, folds, metric=args.metric)
    study_name = args.study or f"{features.name}__{args.model}"
    args.storage_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{args.storage_dir / STUDY_DB}"
    study = tuner.run(args.trials, args.timeout, storage, study_name, args.seed)
    best = study.best_trial
    print(
        f"best {args.metric} = {best.value:.4f} "
        f"(accuracy {best.user_attrs.get('accuracy', float('nan')):.4f}) "
        f"after {len(study.trials)} trials: {best.params}"
    )
    out = save_best(study, tuner, features.name, args.out / f"{study_name}.json")
    print(f"parameters written to {out}")
    return 0


def predict(args: argparse.Namespace) -> int:
    """Predict the test battles with a saved run and write a submission."""
    pokedex = Pokedex.from_json(args.pokedex)
    test = Dataset.from_jsonl(args.test, pokedex, repair=_repair(args))
    features = feature_sets.build(TrainedRun.saved_feature_set(args.run))
    run = TrainedRun.load(args.run, features, sample_battles=test.contexts[:5])
    proba = run.predict_proba(test.contexts)
    submission = pd.DataFrame(
        {
            SUBMISSION_COLUMNS[0]: test.ids,
            SUBMISSION_COLUMNS[1]: (proba > args.threshold).astype(int),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    print(f"{len(submission)} predictions written to {args.out} ({run.name})")
    if args.proba:
        submission.assign(p_player_won=proba).to_csv(args.proba, index=False)
        print(f"probabilities written to {args.proba}")
    return 0


# ----------------------------------------------------------------------
# wiring helpers
# ----------------------------------------------------------------------
def _feature_sets(
    names: Sequence[str], turns: Sequence[int] | None
) -> list[FeatureSet[BattleContext]]:
    """Every named set, once per requested turn (or at the last turn)."""
    sampled: Sequence[int | None] = turns or [None]
    return [feature_sets.build(name, turn) for name in names for turn in sampled]


def _accuracy_by_turn(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean accuracy with turns as rows and ``set / model`` as columns."""
    parsed = summary["feature_set"].map(feature_sets.parse_name)
    table = summary.assign(
        turn=[turn for _, turn in parsed],
        series=[
            f"{base} / {model}"
            for (base, _), model in zip(parsed, summary["model"], strict=True)
        ],
    )
    return table.pivot(index="turn", columns="series", values="accuracy_mean")


def _repair(args: argparse.Namespace) -> TimelineRepair:
    """Fix every replayed battle for P2 knockouts the records omit.

    Always applied, in every command, so that training and prediction see
    the same corrected battles. The detector is learned without labels on
    the training battles only (``predict`` uses the default training file):
    the test battles are only predicted, never used to fit anything, and a
    detector fitted on train + test at prediction time would differ from
    the one the saved model was trained with.
    """
    train = getattr(args, "train", None) or DEFAULT_TRAIN
    return _detector(Path(args.pokedex), Path(train), None)


@cache
def _detector(pokedex_path: Path, train: Path, test: Path | None) -> TimelineRepair:
    pokedex = Pokedex.from_json(pokedex_path)
    return detector_from_files(pokedex, feature_sets.type_chart(), (train, test))


def _report_fold(done: int, total: int, seconds: float) -> None:
    print(f"  fold {done}/{total} ready ({seconds:.0f} s)", flush=True)


def _models(names: Sequence[str], params_files: Sequence[Path] | None) -> list[Model]:
    """Catalog models; with ``--params`` each tuned model (alone or inside an
    ensemble) gets the parameters from its file."""
    tuned = dict(load_params(path) for path in params_files or [])
    for model in tuned:
        if not any(
            model == name
            or model in ENSEMBLES.get(name, ())
            for name in names
        ):
            print(f"note: ignoring tuned parameters for {model!r} (not in --models)")
    return [build_tuned(name, tuned) for name in names]


def _unlabeled(args: argparse.Namespace, pokedex: Pokedex) -> tuple[BattleContext, ...]:
    if args.test is None:
        return ()
    return Dataset.from_jsonl(args.test, pokedex, repair=_repair(args)).contexts


# ----------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pokémon Gen 2 battle prediction")
    parser.add_argument("--pokedex", type=Path, default=DEFAULT_POKEDEX)
    parser.add_argument("-v", "--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    p_check = commands.add_parser("check", help="inspect the data")
    p_check.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p_check.add_argument("--test", type=Path)
    p_check.set_defaults(handler=check)

    for name, handler, help_text in (
        ("evaluate", evaluate, "cross-validate feature sets x models"),
        ("train", train, "fit on all training data and save the runs"),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
        sub.add_argument("--test", type=Path, help="unlabelled battles (optional)")
        sub.add_argument(
            "--features",
            nargs="+",
            required=True,
            choices=sorted(feature_sets.FEATURE_SETS),
        )
        sub.add_argument(
            "--models",
            nargs="+",
            required=True,
            choices=sorted(MODELS),
        )
        sub.add_argument(
            "--turns",
            nargs="+",
            type=int,
            help="sample the features at these turns (default: last turn)",
        )
        sub.add_argument(
            "--params", type=Path, nargs="+", help="tuned parameter files (JSON)"
        )
        sub.set_defaults(handler=handler)
    p_eval = commands.choices["evaluate"]
    p_eval.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    p_eval.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_eval.add_argument(
        "--n-seeds", type=int, default=1,
        help="repeat the CV on this many fold assignments (seeds --seed, --seed+1, ...)",
    )
    p_eval.add_argument("--seeds", type=int, nargs="+", help="explicit seeds (override --n-seeds)")
    p_eval.add_argument(
        "--random-seeds", action="store_true",
        help="draw the --n-seeds seeds at random (they are printed and exported)",
    )
    p_eval.add_argument("--jobs", type=int, default=1, help="seeds evaluated in parallel")
    p_eval.add_argument(
        "--reference",
        help="FEATURE_SET/MODEL: print every other combination's paired difference "
        "with it (corrected repeated k-fold t-test)",
    )
    p_eval.add_argument("--out", type=Path, help="directory for the results")
    commands.choices["train"].add_argument("--out", type=Path, default=Path("runs"))

    p_tune = commands.add_parser("tune", help="search hyperparameters (Optuna)")
    p_tune.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p_tune.add_argument("--test", type=Path, help="unlabelled battles (optional)")
    p_tune.add_argument("--features", required=True, help="one feature set")
    p_tune.add_argument(
        "--model", required=True, choices=sorted(search_spaces())
    )
    p_tune.add_argument("--trials", type=int, default=100)
    p_tune.add_argument("--timeout", type=float, help="seconds (optional)")
    p_tune.add_argument(
        "--metric", choices=["log_loss", "accuracy"], default="log_loss"
    )
    p_tune.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    p_tune.add_argument("--seed", type=int, default=TUNE_SEED)
    p_tune.add_argument("--study", help="study name (default <set>__<model>)")
    p_tune.add_argument("--storage-dir", type=Path, default=Path("runs"))
    p_tune.add_argument("--out", type=Path, default=Path("runs/tuning"))
    p_tune.add_argument(
        "--refresh", action="store_true", help="rebuild the cached fold features"
    )
    p_tune.set_defaults(handler=tune)

    p_predict = commands.add_parser("predict", help="write a submission")
    p_predict.add_argument("--run", type=Path, required=True)
    p_predict.add_argument("--test", type=Path, default=DEFAULT_TEST)
    p_predict.add_argument("--out", type=Path, default=Path("submission.csv"))
    p_predict.add_argument("--proba", type=Path, help="also write probabilities")
    p_predict.add_argument("--threshold", type=float, default=0.5)
    p_predict.set_defaults(handler=predict)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
