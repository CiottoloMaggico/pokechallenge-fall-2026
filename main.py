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

``select`` picks columns one at a time with the corrected repeated k-fold t-test
(``selection.py``): ``--method forward`` (default) adds columns of ``--pool`` to those of
``--start`` and drops useless ones, ``--method backward`` only drops columns of ``--start``
(the revalidation); the folds are ``--repeats`` k-folds seeded ``--seed``, ``--seed`` + 11, ...
and the report goes to ``results/selections/<method>_<start>__<model>.json``::

    python main.py select --start compact_v3 --pool selection_pool --model logreg
    python main.py select --method backward --start compact_v5_orig --repeats 2

``--test`` in ``evaluate``, ``train`` and ``select`` is optional: its battles (never
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

``--repair none | rule | model=<artifact folder> | v11=<artifact folder>`` picks the repair of
the hidden P2 knockouts (default: the detector artifact ``results/addons/knockouts/v3``); a run
saves its repair and ``predict`` refuses another one. The detector is trained, compared and
tuned on its own task (rows = P1's switches) and saved as a new artifact version; v11's
detector has its own artifact (``detector train-v11``)::

    python main.py detector evaluate --features ko_domain30 ko_raw50 --models hgb_ko \\
        --seeds 42 1 2 --reference ko_raw50/hgb_ko
    python main.py detector train --features ko_domain30 --models hgb_ko --version v3
    python main.py detector tune --features ko_domain30 --model hgb_ko --trials 50
    python main.py detector train-v11 --version v1 --seeds 42 1 2
    python main.py replay --battle 42 --repair rule
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections.abc import Sequence
from pathlib import Path

import detector
import usecases
from addons import check_spec
from addons.knockouts_model import ArtifactError
from artifacts import RepairMismatchError
from domain.errors import DomainError
from features.sets import FEATURE_SETS
from features.transitions import TRANSITION_SETS
from folds import FoldPlan
from models import MODELS, search_spaces
from settings import DEFAULT_POKEDEX, DEFAULT_REPAIR, DEFAULT_TEST, DEFAULT_TRAIN, Settings

DEFAULT_FOLDS = 5
DEFAULT_SEED = 42
TUNE_SEED = 2027  # tuning folds differ from evaluation folds
DETECTOR_SEEDS = 3
SELECT_SEED = 11
SELECT_SEED_STEP = 11  # the seeds of the old selection.py: 11, 22, 33
USER_ERRORS = (DomainError, ArtifactError, RepairMismatchError)


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


def _repair(spec: str) -> str:
    try:
        return check_spec(spec)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _check(args: argparse.Namespace) -> int:
    return usecases.check(Settings(args.train, args.test, args.pokedex, repair=args.repair))


def _replay(args: argparse.Namespace) -> int:
    settings = Settings(pokedex=args.pokedex, repair=args.repair)
    return usecases.replay(settings, args.file, args.battle)


def _evaluate(args: argparse.Namespace) -> int:
    plan = FoldPlan(args.folds, _seeds(args))
    settings = Settings(args.train, args.test, args.pokedex, plan, args.jobs, repair=args.repair)
    return usecases.evaluate(
        settings, args.features, args.models, args.turns, args.params, args.reference, args.out
    )


def _train(args: argparse.Namespace) -> int:
    settings = Settings(args.train, args.test, args.pokedex, repair=args.repair)
    return usecases.train(settings, args.features, args.models, args.turns, args.params, args.out)


def _tune(args: argparse.Namespace) -> int:
    plan = FoldPlan(args.folds, (args.seed,))
    settings = Settings(args.train, args.test, args.pokedex, plan, repair=args.repair)
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


def _select(args: argparse.Namespace) -> int:
    seeds = tuple(args.seed + SELECT_SEED_STEP * k for k in range(args.repeats))
    plan = FoldPlan(args.folds, seeds)
    settings = Settings(args.train, args.test, args.pokedex, plan, repair=args.repair)
    thresholds = (args.add_p, args.drop_t, args.p_keep)
    return usecases.select(
        settings, args.start, args.pool, args.model, args.method, thresholds, args.out
    )


def _predict(args: argparse.Namespace) -> int:
    settings = Settings(DEFAULT_TRAIN, args.test, args.pokedex, repair=args.repair)
    return usecases.predict(settings, args.run, args.out, args.proba, args.threshold)


def _detector_evaluate(args: argparse.Namespace) -> int:
    plan = FoldPlan(args.folds, _seeds(args))
    settings = Settings(args.train, pokedex=args.pokedex, plan=plan, jobs=args.jobs)
    return detector.evaluate(
        settings,
        args.features,
        args.models,
        args.params,
        args.reference,
        args.threshold,
        args.agreement,
        args.out,
    )


def _detector_train(args: argparse.Namespace) -> int:
    plan = FoldPlan(args.folds, _seeds(args))
    settings = Settings(args.train, pokedex=args.pokedex, plan=plan, jobs=args.jobs)
    return detector.train(
        settings,
        args.features,
        args.models,
        args.version,
        args.threshold,
        args.params,
        args.agreement,
    )


def _detector_train_v11(args: argparse.Namespace) -> int:
    plan = FoldPlan(args.folds, _seeds(args))
    settings = Settings(args.train, pokedex=args.pokedex, plan=plan)
    return detector.train_v11(settings, args.version)


def _detector_tune(args: argparse.Namespace) -> int:
    settings = Settings(args.train, pokedex=args.pokedex, plan=FoldPlan(args.folds, (args.seed,)))
    return detector.tune(
        settings,
        args.features,
        args.model,
        args.trials,
        args.timeout,
        args.metric,
        args.study,
        args.storage_dir,
        args.out,
    )


def _add_repair(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--repair",
        type=_repair,
        default=DEFAULT_REPAIR,
        help=f"none, rule, model=<artifact folder> or v11=<artifact folder> "
        f"(default {DEFAULT_REPAIR})",
    )


def _add_seeds(sub: argparse.ArgumentParser, n_seeds: int) -> None:
    sub.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    sub.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sub.add_argument(
        "--n-seeds",
        type=int,
        default=n_seeds,
        help="repeat the CV on this many fold assignments (seeds --seed, --seed+1, ...)",
    )
    sub.add_argument("--seeds", type=int, nargs="+", help="explicit seeds (override --n-seeds)")
    sub.add_argument(
        "--random-seeds",
        action="store_true",
        help="draw the --n-seeds seeds at random (they are printed and exported)",
    )
    sub.add_argument("--jobs", type=int, default=1, help="seeds evaluated in parallel")


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
        _add_repair(sub)
        sub.set_defaults(handler=handler)
    p_eval = commands.choices["evaluate"]
    _add_seeds(p_eval, 1)
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


def _add_tune(commands: argparse._SubParsersAction, name: str, sets: Sequence[str] | None) -> None:
    p_tune = commands.add_parser(name, help="search hyperparameters (Optuna)")
    p_tune.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p_tune.add_argument("--features", required=True, choices=sets, help="one feature set")
    p_tune.add_argument("--model", required=True, choices=sorted(search_spaces()))
    p_tune.add_argument("--trials", type=int, default=100)
    p_tune.add_argument("--timeout", type=float, help="seconds (optional)")
    p_tune.add_argument("--metric", choices=["log_loss", "accuracy"], default="log_loss")
    p_tune.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    p_tune.add_argument("--seed", type=int, default=TUNE_SEED)
    p_tune.add_argument("--study", help="study name (default <set>__<model>)")
    p_tune.add_argument("--storage-dir", type=Path, help="study database (default results/tuning)")
    p_tune.add_argument("--out", type=Path, help="best parameters (default results/tuning)")


def _add_select(commands: argparse._SubParsersAction) -> None:
    p_select = commands.add_parser("select", help="select columns (corrected t-test)")
    p_select.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p_select.add_argument("--test", type=Path, help="unlabelled battles (optional)")
    p_select.add_argument("--method", choices=["forward", "backward"], default="forward")
    p_select.add_argument("--start", default="compact_v3", help="feature set to start from")
    p_select.add_argument("--pool", default="selection_pool", help="candidates of forward")
    p_select.add_argument("--model", default="logreg", choices=sorted(MODELS))
    p_select.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    p_select.add_argument("--repeats", type=int, default=3, help="k-folds with different seeds")
    p_select.add_argument("--seed", type=int, default=SELECT_SEED)
    p_select.add_argument("--add-p", type=float, default=0.05, help="forward: corrected p to add")
    p_select.add_argument("--drop-t", type=float, default=1.0, help="forward: corrected t to drop")
    p_select.add_argument(
        "--p-keep", type=float, default=0.20, help="backward: one-sided p to drop a column"
    )
    p_select.add_argument("--out", type=Path, help="report (default results/selections/)")
    _add_repair(p_select)
    p_select.set_defaults(handler=_select)


def _add_detector(commands: argparse._SubParsersAction) -> None:
    group = commands.add_parser("detector", help="the knockout detector (rows = P1's switches)")
    detector_commands = group.add_subparsers(dest="detector_command", required=True)
    sets = sorted(TRANSITION_SETS)
    p_eval = detector_commands.add_parser("evaluate", help="cross-validate detectors + checks")
    p_eval.add_argument("--features", nargs="+", required=True, choices=sets)
    p_eval.add_argument("--models", nargs="+", required=True, choices=sorted(MODELS))
    p_eval.add_argument("--reference", help="FEATURE_SET/MODEL for the paired differences")
    p_eval.add_argument("--out", type=Path, help="default results/evaluations/detector_<date>")
    p_eval.set_defaults(handler=_detector_evaluate)
    p_train = detector_commands.add_parser("train", help="save a new detector artifact version")
    p_train.add_argument("--features", default="ko_domain30", choices=sets)
    p_train.add_argument("--models", default="hgb_ko", choices=sorted(MODELS))
    p_train.add_argument("--version", required=True, help="results/addons/knockouts/<version>")
    p_train.set_defaults(handler=_detector_train)
    for sub in (p_eval, p_train):
        sub.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
        sub.add_argument("--params", type=Path, nargs="+", help="tuned parameter files (JSON)")
        sub.add_argument("--threshold", type=float, default=0.5, help="P(faint) of a repair")
        sub.add_argument("--agreement", type=Path, help="artifact folder to compare decisions with")
        _add_seeds(sub, DETECTOR_SEEDS)
    p_v11 = detector_commands.add_parser("train-v11", help="save a new artifact of v11's detector")
    p_v11.add_argument("--version", required=True, help="results/addons/knockouts_v11/<version>")
    p_v11.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    _add_seeds(p_v11, DETECTOR_SEEDS)
    p_v11.set_defaults(handler=_detector_train_v11)
    _add_tune(detector_commands, "tune", sets)
    detector_commands.choices["tune"].set_defaults(handler=_detector_tune)


def build_parser() -> argparse.ArgumentParser:
    """The parser of every command; each one sets ``handler``."""
    parser = argparse.ArgumentParser(description="Pokémon Gen 2 battle prediction")
    parser.add_argument("--pokedex", type=Path, default=DEFAULT_POKEDEX)
    parser.add_argument("-v", "--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    p_check = commands.add_parser("check", help="inspect the data")
    p_check.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p_check.add_argument("--test", type=Path)
    _add_repair(p_check)
    p_check.set_defaults(handler=_check)

    p_replay = commands.add_parser("replay", help="print the repaired states of one battle")
    p_replay.add_argument("--battle", type=int, required=True)
    p_replay.add_argument("--file", type=Path, default=DEFAULT_TRAIN)
    _add_repair(p_replay)
    p_replay.set_defaults(handler=_replay)

    _add_evaluate_and_train(commands)
    _add_tune(commands, "tune", None)  # any set, @t<N> included
    p_tune = commands.choices["tune"]
    p_tune.add_argument("--test", type=Path, help="unlabelled battles (optional)")
    p_tune.add_argument("--refresh", action="store_true", help="rebuild the cached fold features")
    _add_repair(p_tune)
    p_tune.set_defaults(handler=_tune)
    _add_select(commands)
    _add_detector(commands)

    p_predict = commands.add_parser("predict", help="write a submission")
    p_predict.add_argument("--run", type=Path, required=True)
    p_predict.add_argument("--test", type=Path, default=DEFAULT_TEST)
    p_predict.add_argument(
        "--out", type=Path, help="submission file (default results/submissions/submission.csv)"
    )
    p_predict.add_argument("--proba", type=Path, help="also write probabilities")
    p_predict.add_argument("--threshold", type=float, default=0.5)
    _add_repair(p_predict)
    p_predict.set_defaults(handler=_predict)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command; a domain, artifact or repair error is one line and exit code 2."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.handler(args))
    except USER_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
