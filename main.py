"""Command line: parses the arguments, builds the settings, calls one use case.

Commands::

    python main.py check    --train input/train.jsonl --test input/test.jsonl
    python main.py evaluate --train input/train.jsonl --features hp30 \\
        --models logreg hgb ensemble --test input/test.jsonl --out results/evaluations/hp30
    python main.py train    --train input/train.jsonl --features hp30 \\
        --models ensemble --test input/test.jsonl
    python main.py predict  --run results/runs/hp30__ensemble --test input/test.jsonl
    python main.py evaluate --train input/train.jsonl --features core \\
        --models logreg --turns 5 10 15 20 25 30 --out results/evaluations/by_turn
    python main.py tune     --features compact --model hgb --trials 200
    python main.py evaluate --features compact --models hgb \\
        --params results/tuning/compact__hgb.json

``--test`` in ``evaluate`` and ``train`` is optional: its battles (never
its labels, which it does not have) feed label-free stateful features.
Everything is written under ``results/`` unless ``--out`` says otherwise.

``tune`` searches a model's hyperparameters with Optuna (``pip install
optuna``) on folds seeded differently from ``evaluate``, stores the study
in SQLite (several terminals can share it) and writes the best
parameters to ``results/tuning/<set>__<model>.json``; ``--params`` (one or
more files) then builds those models with them, also inside ensembles
(named ``<model>-tuned``).

``evaluate --n-seeds 5 --jobs 2`` repeats the cross-validation on five
fold assignments (seeds 42..46; ``--seeds 1 7 99`` picks them,
``--random-seeds`` draws them) and adds ``accuracy_seed_std`` and
``log_loss_seed_std``, how much the result moves with the luck of the
folds, plus an accuracy-by-seed table; the seeds are in ``meta.json`` and
in the ``seed`` column of ``folds.csv`` and ``predictions.csv``. ``--jobs``
is ignored, with a message, for models and features that use LightGBM,
XGBoost or CatBoost.

``--turns`` samples every feature set at each given turn (sets named
``core@t5``, ``core@t10``, ...); ``evaluate`` then also prints and
exports ``accuracy_by_turn.csv``, ready for an accuracy-vs-turn plot.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections.abc import Sequence
from pathlib import Path

import usecases
from domain.errors import DomainError
from features.sets import FEATURE_SETS
from folds import FoldPlan
from models import MODELS, search_spaces
from settings import DEFAULT_POKEDEX, DEFAULT_TEST, DEFAULT_TRAIN, Settings

DEFAULT_FOLDS = 5
DEFAULT_SEED = 42
TUNE_SEED = 2027  # tuning folds differ from evaluation folds


def _seeds(args: argparse.Namespace) -> tuple[int, ...]:
    """The seeds of the fold assignments: ``--seeds`` as given, else
    ``--n-seeds`` random ones (``--random-seeds``) or consecutive ones
    starting from ``--seed`` (default: just ``--seed``)."""
    if args.seeds:
        return tuple(dict.fromkeys(args.seeds))
    if args.n_seeds < 1:
        raise SystemExit("--n-seeds must be at least 1")
    if args.random_seeds:
        rng = random.SystemRandom()
        return tuple(rng.sample(range(1, 2**31 - 1), args.n_seeds))
    return tuple(args.seed + k for k in range(args.n_seeds))


def _check(args: argparse.Namespace) -> int:
    return usecases.check(Settings(args.train, args.test, args.pokedex))


def _evaluate(args: argparse.Namespace) -> int:
    plan = FoldPlan(args.folds, _seeds(args))
    settings = Settings(args.train, args.test, args.pokedex, plan, args.jobs)
    return usecases.evaluate(
        settings, args.features, args.models, args.turns, args.params, args.reference, args.out
    )


def _train(args: argparse.Namespace) -> int:
    settings = Settings(args.train, args.test, args.pokedex)
    return usecases.train(settings, args.features, args.models, args.turns, args.params, args.out)


def _tune(args: argparse.Namespace) -> int:
    plan = FoldPlan(args.folds, (args.seed,))
    settings = Settings(args.train, args.test, args.pokedex, plan)
    return usecases.tune(
        settings,
        args.features,
        args.model,
        args.trials,
        args.timeout,
        args.metric,
        args.study,
        args.storage_dir,
        args.out,
        args.refresh,
    )


def _predict(args: argparse.Namespace) -> int:
    settings = Settings(DEFAULT_TRAIN, args.test, args.pokedex)
    return usecases.predict(settings, args.run, args.out, args.proba, args.threshold)


def _add_evaluate_and_train(commands: argparse._SubParsersAction) -> None:
    for name, handler, help_text in (
        ("evaluate", _evaluate, "cross-validate feature sets x models"),
        ("train", _train, "fit on all training data and save the runs"),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
        sub.add_argument("--test", type=Path, help="unlabelled battles (optional)")
        sub.add_argument("--features", nargs="+", required=True, choices=sorted(FEATURE_SETS))
        sub.add_argument("--models", nargs="+", required=True, choices=sorted(MODELS))
        sub.add_argument(
            "--turns",
            nargs="+",
            type=int,
            help="sample the features at these turns (default: last turn)",
        )
        sub.add_argument("--params", type=Path, nargs="+", help="tuned parameter files (JSON)")
        sub.set_defaults(handler=handler)
    p_eval = commands.choices["evaluate"]
    p_eval.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    p_eval.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_eval.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="repeat the CV on this many fold assignments (seeds --seed, --seed+1, ...)",
    )
    p_eval.add_argument("--seeds", type=int, nargs="+", help="explicit seeds (override --n-seeds)")
    p_eval.add_argument(
        "--random-seeds",
        action="store_true",
        help="draw the --n-seeds seeds at random (they are printed and exported)",
    )
    p_eval.add_argument("--jobs", type=int, default=1, help="seeds evaluated in parallel")
    p_eval.add_argument(
        "--reference",
        help="FEATURE_SET/MODEL: print every other combination's paired difference "
        "with it (corrected repeated k-fold t-test)",
    )
    p_eval.add_argument(
        "--out", type=Path, help="directory for the results (default results/evaluations/<date>)"
    )
    commands.choices["train"].add_argument(
        "--out", type=Path, help="parent of the run folders (default results/runs)"
    )


def _add_tune(commands: argparse._SubParsersAction) -> None:
    p_tune = commands.add_parser("tune", help="search hyperparameters (Optuna)")
    p_tune.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p_tune.add_argument("--test", type=Path, help="unlabelled battles (optional)")
    p_tune.add_argument("--features", required=True, help="one feature set")
    p_tune.add_argument("--model", required=True, choices=sorted(search_spaces()))
    p_tune.add_argument("--trials", type=int, default=100)
    p_tune.add_argument("--timeout", type=float, help="seconds (optional)")
    p_tune.add_argument("--metric", choices=["log_loss", "accuracy"], default="log_loss")
    p_tune.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    p_tune.add_argument("--seed", type=int, default=TUNE_SEED)
    p_tune.add_argument("--study", help="study name (default <set>__<model>)")
    p_tune.add_argument("--storage-dir", type=Path, help="study database (default results/tuning)")
    p_tune.add_argument("--out", type=Path, help="best parameters (default results/tuning)")
    p_tune.add_argument("--refresh", action="store_true", help="rebuild the cached fold features")
    p_tune.set_defaults(handler=_tune)


def build_parser() -> argparse.ArgumentParser:
    """The parser of every command; each one sets ``handler``."""
    parser = argparse.ArgumentParser(description="Pokémon Gen 2 battle prediction")
    parser.add_argument("--pokedex", type=Path, default=DEFAULT_POKEDEX)
    parser.add_argument("-v", "--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    p_check = commands.add_parser("check", help="inspect the data")
    p_check.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p_check.add_argument("--test", type=Path)
    p_check.set_defaults(handler=_check)

    _add_evaluate_and_train(commands)
    _add_tune(commands)

    p_predict = commands.add_parser("predict", help="write a submission")
    p_predict.add_argument("--run", type=Path, required=True)
    p_predict.add_argument("--test", type=Path, default=DEFAULT_TEST)
    p_predict.add_argument(
        "--out", type=Path, help="submission file (default results/submissions/submission.csv)"
    )
    p_predict.add_argument("--proba", type=Path, help="also write probabilities")
    p_predict.add_argument("--threshold", type=float, default=0.5)
    p_predict.set_defaults(handler=_predict)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command; a domain error becomes a one-line message and exit code 2."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.handler(args))
    except DomainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
