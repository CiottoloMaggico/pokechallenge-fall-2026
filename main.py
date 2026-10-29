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
repaired for the P2 knockouts the records omit (``domain.knockouts``).

``tune`` searches a model's hyperparameters with Optuna (``pip install
optuna``) on folds seeded differently from ``evaluate``, stores the study
in SQLite (several terminals can share it) and writes the best
parameters to ``runs/tuning/<set>__<model>.json``; ``--params`` (one or
more files) then builds those models with them, also inside ensembles
(named ``<model>-tuned``).

``--turns`` samples every feature set at each given turn (sets named
``core@t5``, ``core@t10``, ...); ``evaluate`` then also prints and
exports ``accuracy_by_turn.csv``, ready for an accuracy-vs-turn plot.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import feature_catalog
import model_catalog
from data import Dataset, Repair
from domain.battle import BattleContext
from domain.pokedex import Pokedex
from feature_set import FeatureSet
from gym import FoldData, Gym, TrainedRun, prepare_folds, save_runs
from knockout_detector import KnockoutDetector, detector_from_files
from model import Model
from models.ensemble import DEFAULT_MEMBERS
from tuning import Tuner, cached_folds, load_params, save_best

DEFAULT_POKEDEX = Path("assets/species.json")
DEFAULT_TRAIN = Path("input/train.jsonl")
DEFAULT_TEST = Path("input/test.jsonl")
DEFAULT_FOLDS = 5
DEFAULT_SEED = 42
TUNE_SEED = 2027  # tuning folds differ from evaluation folds
STUDY_DB = "optuna.db"
SUBMISSION_COLUMNS = ("battle_id", "player_won")

logger = logging.getLogger("main")


# ----------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------
def check(args: argparse.Namespace) -> int:
    """Load the data and report what the pipeline will do with it."""
    pokedex = Pokedex.from_json(args.pokedex)
    train = Dataset.from_jsonl(args.train, pokedex, repair=_repair(args))
    groups = train.duplicate_groups()
    print(f"pokedex: {len(pokedex)} species")
    print(
        f"train: {len(train)} battles, P1 win rate {train.labels.mean():.3f}, "
        f"{len(train) - len(set(groups))} duplicates, "
        f"{len(train.substitutions)} species replaced"
    )
    if args.test:
        test = Dataset.from_jsonl(args.test, pokedex, repair=_repair(args))
        print(
            f"test: {len(test)} battles, labelled {test.is_labelled}, "
            f"{len(test.substitutions)} species replaced"
        )
    return 0


def evaluate(args: argparse.Namespace) -> int:
    """Cross-validate every feature set with every model and export."""
    pokedex = Pokedex.from_json(args.pokedex)
    train = Dataset.from_jsonl(args.train, pokedex, repair=_repair(args))
    unlabeled = _unlabeled(args, pokedex)
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    feature_sets = _feature_sets(args.features, args.turns)
    gym = Gym(_models(args.models, args.params), feature_sets, splitter)
    result = gym.evaluate(
        train.contexts,
        train.labels,
        ids=train.ids,
        groups=train.duplicate_groups(),
        unlabeled=unlabeled,
    )
    summary = result.summary().sort_values("accuracy_mean", ascending=False)
    print(summary.to_string(index=False, float_format="{:.4f}".format))
    by_turn = _accuracy_by_turn(summary) if args.turns else None
    if by_turn is not None:
        print(by_turn.to_string(float_format="{:.4f}".format))
    if args.out:
        result.export(args.out)
        if by_turn is not None:
            by_turn.to_csv(args.out / "accuracy_by_turn.csv")
        print(f"results exported to {args.out}")
    return 0


def train(args: argparse.Namespace) -> int:
    """Fit every combination on all training data and save the runs."""
    pokedex = Pokedex.from_json(args.pokedex)
    dataset = Dataset.from_jsonl(args.train, pokedex, repair=_repair(args))
    gym = Gym(
        _models(args.models, args.params), _feature_sets(args.features, args.turns)
    )
    runs = gym.train(
        dataset.contexts,
        dataset.labels,
        unlabeled=_unlabeled(args, pokedex),
        groups=dataset.duplicate_groups(),
    )
    for path in save_runs(runs, args.out):
        print(f"saved {path}")
    return 0


def tune(args: argparse.Namespace) -> int:
    """Search the hyperparameters of one model on one feature set."""
    features = feature_catalog.build(args.features)
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
    features = feature_catalog.build(TrainedRun.saved_feature_set(args.run))
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
    return [feature_catalog.build(name, turn) for name in names for turn in sampled]


def _accuracy_by_turn(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean accuracy with turns as rows and ``set / model`` as columns."""
    parsed = summary["feature_set"].map(feature_catalog.parse_name)
    table = summary.assign(
        turn=[turn for _, turn in parsed],
        series=[
            f"{base} / {model}"
            for (base, _), model in zip(parsed, summary["model"], strict=True)
        ],
    )
    return table.pivot(index="turn", columns="series", values="accuracy_mean")


def _repair(args: argparse.Namespace) -> Repair:
    """Fix every replayed battle for P2 knockouts the records omit.

    Always applied, in every command, so that training and prediction see
    the same corrected battles. The detector is learned without labels on
    the training battles plus the test battles when given (``predict``
    uses the default training file).
    """
    train = getattr(args, "train", None) or DEFAULT_TRAIN
    return _detector(Path(args.pokedex), Path(train), getattr(args, "test", None))


@cache
def _detector(pokedex_path: Path, train: Path, test: Path | None) -> KnockoutDetector:
    pokedex = Pokedex.from_json(pokedex_path)
    return detector_from_files(pokedex, feature_catalog.type_chart(), (train, test))


def _report_fold(done: int, total: int, seconds: float) -> None:
    print(f"  fold {done}/{total} ready ({seconds:.0f} s)", flush=True)


def _models(names: Sequence[str], params_files: Sequence[Path] | None) -> list[Model]:
    """Catalog models; with ``--params`` each tuned model (alone or inside an
    ensemble) gets the parameters from its file."""
    tuned = dict(load_params(path) for path in params_files or [])
    for model in tuned:
        if not any(
            model == name
            or model in model_catalog.ENSEMBLES.get(name, ())
            or (name == "ensemble" and model in DEFAULT_MEMBERS)
            for name in names
        ):
            print(f"note: ignoring tuned parameters for {model!r} (not in --models)")
    return [model_catalog.build_tuned(name, tuned) for name in names]


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
            choices=sorted(feature_catalog.FEATURE_SETS),
        )
        sub.add_argument(
            "--models",
            nargs="+",
            required=True,
            choices=sorted(model_catalog.BUILDERS),
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
    p_eval.add_argument("--out", type=Path, help="directory for the results")
    commands.choices["train"].add_argument("--out", type=Path, default=Path("runs"))

    p_tune = commands.add_parser("tune", help="search hyperparameters (Optuna)")
    p_tune.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p_tune.add_argument("--test", type=Path, help="unlabelled battles (optional)")
    p_tune.add_argument("--features", required=True, help="one feature set")
    p_tune.add_argument(
        "--model", required=True, choices=sorted(model_catalog.SEARCH_SPACES)
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
