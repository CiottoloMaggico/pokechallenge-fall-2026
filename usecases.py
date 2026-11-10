"""What each command of ``main.py`` does, given its settings.

Every battle goes through the repair of ``Settings.repair`` (``addons``: by default the
detector artifact of the hidden P2 knockouts), in every command, so training and prediction
see the same corrected battles; a run refuses to predict with another repair. Duplicated
battles are always kept in the same fold.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import pandas as pd

import domain.__main__ as domain_cli
from addons import load_repair
from artifacts import TrainedRun
from data import Dataset
from domain.battle import BattleContext
from domain.pokedex import Pokedex, species_without_stats
from domain.repair import TimelineRepair
from feature_set import FeatureSet
from features import sets as feature_sets
from folds import FoldData, read_fold_tables
from gym import Gym, GymResult
from interfaces import Model
from models import ENSEMBLES, MODELS, build_tuned
from registry import fork_unsafe_libraries
from selection import Selector
from settings import DEFAULT_TEST, Settings
from tuning.params import load_params
from tuning.tuner import Tuner, save_best

STUDY_DB = "optuna.db"
SUBMISSION_COLUMNS = ("battle_id", "player_won")
MAX_SEED_COLUMNS = 10  # more seeds: the accuracy-by-seed table is only exported


def check(settings: Settings) -> int:
    """Load the data and report what the pipeline will do with it."""
    pokedex = Pokedex.from_json(settings.pokedex)
    train = Dataset.from_jsonl(settings.train, pokedex, repair=_repair(settings))
    groups = train.duplicate_groups()
    print(f"pokedex: {len(pokedex)} species (source sha256 {pokedex.source_sha256})")
    print(
        f"train: {len(train)} battles, P1 win rate {train.labels.mean():.3f}, "
        f"{len(train) - len(set(groups))} duplicates, "
        f"{_without_stats(train, pokedex)}"
    )
    if settings.test:
        test = Dataset.from_jsonl(settings.test, pokedex, repair=_repair(settings))
        print(
            f"test: {len(test)} battles, labelled {test.is_labelled}, "
            f"{_without_stats(test, pokedex)}"
        )
    return 0


def _without_stats(dataset: Dataset, pokedex: Pokedex) -> str:
    missing = species_without_stats((c.record for c in dataset), pokedex)
    return f"species without stats: {missing or 'none'}"


def evaluate(
    settings: Settings,
    features: Sequence[str],
    models: Sequence[str],
    turns: Sequence[int] | None = None,
    params: Sequence[Path] | None = None,
    reference: str | None = None,
    out: Path | None = None,
) -> int:
    """Cross-validate every feature set with every model on each seed of the plan and export.

    With several seeds the table reports, besides the fold std, the std across seeds of
    accuracy and log loss. Without ``out`` the tables go to a dated evaluation folder.
    """
    _refuse_test_with_v11(settings, features)
    jobs = fork_safe_jobs(settings.jobs, features, models)
    pokedex = Pokedex.from_json(settings.pokedex)
    train = Dataset.from_jsonl(settings.train, pokedex, repair=_repair(settings))
    unlabeled = _unlabeled(settings, pokedex)
    print(f"seeds: {list(settings.plan.seeds)}", flush=True)
    gym = Gym(
        build_models(models, params), _feature_sets(features, turns), settings.plan, jobs=jobs
    )
    result = gym.evaluate(train.labelled(), unlabeled)
    target = out or settings.results.evaluation(time.strftime("evaluate_%Y-%m-%d_%H%M%S"))
    report(result, reference, bool(turns), target)
    return 0


def _refuse_test_with_v11(settings: Settings, features: Sequence[str]) -> None:
    """v11's label-free features fit on the training battles only, never on ``--test``."""
    v11_sets = ("v11", "v11_fs")
    if settings.test is not None and any(
        feature_sets.parse_name(f)[0] in v11_sets for f in features
    ):
        raise SystemExit("v11 fits its features on the training battles only: drop --test")


def fork_safe_jobs(jobs: int, features: Sequence[str], models: Sequence[str]) -> int:
    """``jobs``, or 1 with a message if a selected model or feature hangs in forked workers."""
    libraries = fork_unsafe_libraries(models, features)
    if jobs > 1 and libraries:
        print(
            f"note: --jobs {jobs} ignored: {', '.join(libraries)} use OpenMP, which hangs in "
            "forked workers; seeds run in sequence",
            flush=True,
        )
        return 1
    return jobs


def report(result: GymResult, reference: str | None, by_turn: bool, out: Path) -> None:
    """Print the tables of an evaluation and export them with the raw results."""
    summary = result.summary().sort_values("accuracy_mean", ascending=False)
    print(summary.to_string(index=False, float_format="{:.4f}".format))
    result.export(out)
    summary.to_csv(out / "summary.csv", index=False)
    n_seeds = len(set(result.folds["seed"]))
    if n_seeds > 1:
        by_seed = _accuracy_by_seed(result, summary)
        by_seed.to_csv(out / "accuracy_by_seed.csv")
        if n_seeds <= MAX_SEED_COLUMNS:
            print("\naccuracy by seed (mean over the folds):")
            print(by_seed.to_string(float_format="{:.4f}".format))
    if reference:
        paired = _paired(result, reference)
        paired.to_csv(out / "paired.csv", index=False)
        print(f"\npaired vs {reference} (seed x fold, corrected t-test):")
        print(paired.to_string(index=False, float_format="{:.4f}".format))
    if by_turn:
        turns = _accuracy_by_turn(summary)
        turns.to_csv(out / "accuracy_by_turn.csv")
        print(turns.to_string(float_format="{:.4f}".format))
    print(f"results exported to {out}")


def _accuracy_by_seed(result: GymResult, summary: pd.DataFrame) -> pd.DataFrame:
    """Accuracy of each seed, in the order of the summary."""
    table = result.by_seed().set_index(["feature_set", "model"])
    return table.loc[list(zip(summary["feature_set"], summary["model"], strict=True))]


def _paired(result: GymResult, reference: str) -> pd.DataFrame:
    """Accuracy and log loss of every other combination against ``SET/MODEL``."""
    fs_name, _, model = reference.partition("/")
    paired = pd.concat(
        [
            result.paired((fs_name, model), metric).assign(metric=metric)
            for metric in ("accuracy", "log_loss")
        ],
        ignore_index=True,
    )
    is_reference = (paired["feature_set"] == fs_name) & (paired["model"] == model)
    return paired.loc[~is_reference]


def _accuracy_by_turn(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean accuracy with turns as rows and ``set / model`` as columns."""
    parsed = summary["feature_set"].map(feature_sets.parse_name)
    table = summary.assign(
        turn=[turn for _, turn in parsed],
        series=[
            f"{base} / {model}" for (base, _), model in zip(parsed, summary["model"], strict=True)
        ],
    )
    return table.pivot(index="turn", columns="series", values="accuracy_mean")


def train(
    settings: Settings,
    features: Sequence[str],
    models: Sequence[str],
    turns: Sequence[int] | None = None,
    params: Sequence[Path] | None = None,
    out: Path | None = None,
) -> int:
    """Fit every combination on all training data and save the runs (``out/<set>__<model>``)."""
    _refuse_test_with_v11(settings, features)
    pokedex = Pokedex.from_json(settings.pokedex)
    repair = _repair(settings)
    dataset = Dataset.from_jsonl(settings.train, pokedex, repair=repair)
    gym = Gym(build_models(models, params), _feature_sets(features, turns))
    runs = gym.train(
        dataset.labelled(), unlabeled=_unlabeled(settings, pokedex), repair=repair.describe()
    )
    for run in runs:
        folder = out / run.name if out else settings.results.run(run.feature_set, run.model_name)
        print(f"saved {run.save(folder)}")
    return 0


def tune(
    settings: Settings,
    features: str,
    model: str,
    trials: int,
    timeout: float | None = None,
    metric: str = "log_loss",
    study: str | None = None,
    storage_dir: Path | None = None,
    out: Path | None = None,
    refresh: bool = False,
) -> int:
    """Search the hyperparameters of one model on one feature set, on the first seed of the plan.

    The fold tables are built once and cached (``refresh`` rebuilds them); several
    processes can share the study, stored in SQLite.
    """
    source = feature_sets.build(features)
    plan = settings.plan
    seed = plan.seeds[0]
    tables_dir = settings.results.cache("folds")
    tables = (
        None if refresh else read_fold_tables(tables_dir, source.name, plan, bool(settings.test))
    )
    if tables is None:
        print(f"preparing the features of {plan.n_folds} folds (once, then cached)")
        pokedex = Pokedex.from_json(settings.pokedex)
        train_set = Dataset.from_jsonl(settings.train, pokedex, repair=_repair(settings))
        tables = Gym.fold_tables(
            train_set.labelled(),
            source,
            plan,
            _unlabeled(settings, pokedex),
            cache=tables_dir,
            progress=_report_fold,
        )
    print(f"fold features ready ({tables_dir}); starting the trials")
    tuner = Tuner(MODELS[model], tables[seed], metric=metric)
    study_name = study or f"{source.name}__{model}"
    run_study(settings, tuner, source.name, study_name, trials, timeout, storage_dir, out)
    return 0


def run_study(
    settings: Settings,
    tuner: Tuner,
    feature_set: str,
    study: str,
    trials: int,
    timeout: float | None,
    storage_dir: Path | None,
    out: Path | None,
) -> None:
    """Run (or resume) the Optuna study on the first seed of the plan and save the best trial."""
    storage_root = storage_dir or settings.results.tuning()
    storage_root.mkdir(parents=True, exist_ok=True)
    found = tuner.run(
        trials, timeout, f"sqlite:///{storage_root / STUDY_DB}", study, settings.plan.seeds[0]
    )
    best = found.best_trial
    print(
        f"best {tuner.metric} = {best.value:.4f} "
        f"(accuracy {best.user_attrs.get('accuracy', float('nan')):.4f}) "
        f"after {len(found.trials)} trials: {best.params}"
    )
    target = (out or settings.results.tuning()) / f"{study}.json"
    print(f"parameters written to {save_best(found, tuner, feature_set, target)}")


def select(
    settings: Settings,
    start: str,
    pool: str,
    model: str,
    method: str = "forward",
    thresholds: tuple[float, float, float] = (0.05, 1.0, 0.20),
    out: Path | None = None,
) -> int:
    """Select columns with ``model`` on every fold of the plan and save the report.

    ``backward`` drops columns of ``start`` (tables of ``start``); ``forward`` also adds
    columns of ``pool`` (tables of ``pool``). ``thresholds``: p_add, drop_t, p_keep.
    """
    columns = feature_sets.build(start).declared_columns
    if columns is None:
        raise SystemExit(f"{start!r} does not declare its columns")
    selector = Selector(_selection_folds(settings, start if method == "backward" else pool), model)
    missing = set(columns) - set(selector.columns)
    if missing:
        raise SystemExit(f"start columns not in the pool: {sorted(missing)}")
    p_add, drop_t, p_keep = thresholds
    found = (
        selector.backward(columns, p_keep)
        if method == "backward"
        else selector.forward(columns, selector.columns, p_add, drop_t)
    )
    result = {
        "method": method,
        "start": start,
        "pool": pool if method == "forward" else start,
        "model": model,
        "folds": len(selector.folds),
        **found.to_dict(),
    }
    target = out or settings.results.selection(f"{method}_{start}__{model}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(f"final columns ({len(found.columns)}): {found.columns}")
    print(f"saved to {target}")
    return 0


def _selection_folds(settings: Settings, source: str) -> list[FoldData]:
    """The tables of ``source`` on every fold of every seed of the plan."""
    pokedex = Pokedex.from_json(settings.pokedex)
    train = Dataset.from_jsonl(settings.train, pokedex, repair=_repair(settings))
    began = time.perf_counter()
    tables = Gym.fold_tables(
        train.labelled(),
        feature_sets.build(source),
        settings.plan,
        _unlabeled(settings, pokedex),
        progress=_report_fold,
    )
    print(f"tables of {source} ready in {time.perf_counter() - began:.0f} s", flush=True)
    return [fold for seed in settings.plan.seeds for fold in tables[seed]]


def predict(
    settings: Settings,
    run_dir: Path,
    out: Path | None = None,
    proba: Path | None = None,
    threshold: float = 0.5,
) -> int:
    """Predict the test battles with a saved run and write a submission.

    The run must have been trained with the requested repair (``RepairMismatchError``).
    """
    pokedex = Pokedex.from_json(settings.pokedex)
    repair = _repair(settings)
    features = feature_sets.build(TrainedRun.saved_feature_set(run_dir))
    run = TrainedRun.load(run_dir, features)
    run.check_repair(repair.describe())
    test = Dataset.from_jsonl(settings.test or DEFAULT_TEST, pokedex, repair=repair)
    run.check_columns(test.contexts[:5])
    probabilities = run.predict_proba(test.contexts)
    submission = pd.DataFrame(
        {
            SUBMISSION_COLUMNS[0]: test.ids,
            SUBMISSION_COLUMNS[1]: (probabilities > threshold).astype(int),
        }
    )
    target = out or settings.results.submission("submission")
    target.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(target, index=False)
    print(f"{len(submission)} predictions written to {target} ({run.name})")
    if proba:
        submission.assign(p_player_won=probabilities).to_csv(proba, index=False)
        print(f"probabilities written to {proba}")
    return 0


def _feature_sets(
    names: Sequence[str], turns: Sequence[int] | None
) -> list[FeatureSet[BattleContext]]:
    """Every named set, once per requested turn (or at the last turn)."""
    sampled: Sequence[int | None] = turns or [None]
    return [feature_sets.build(name, turn) for name in names for turn in sampled]


def replay(settings: Settings, path: Path, battle_id: int) -> int:
    """Print the states of one battle of ``path``, repaired with ``settings.repair``."""
    return domain_cli.replay(path, battle_id, settings.pokedex, _repair(settings))


def _repair(settings: Settings) -> TimelineRepair:
    """The repair of ``settings``, loaded once per process so that its cache is shared."""
    return _load_repair(settings.repair, settings.pokedex)


@cache
def _load_repair(spec: str, pokedex_path: Path) -> TimelineRepair:
    return load_repair(spec, Pokedex.from_json(pokedex_path), feature_sets.type_chart())


def _report_fold(done: int, total: int, seconds: float) -> None:
    print(f"  fold {done}/{total} ready ({seconds:.0f} s)", flush=True)


def build_models(names: Sequence[str], params_files: Sequence[Path] | None) -> list[Model]:
    """Registered models; with ``params_files`` each tuned model (alone or inside an
    ensemble) gets the parameters from its file."""
    tuned = dict(load_params(path) for path in params_files or [])
    for model in tuned:
        if not any(model == name or model in ENSEMBLES.get(name, ()) for name in names):
            print(f"note: ignoring tuned parameters for {model!r} (not in --models)")
    return [build_tuned(name, tuned) for name in names]


def _unlabeled(settings: Settings, pokedex: Pokedex) -> tuple[BattleContext, ...]:
    if settings.test is None:
        return ()
    return Dataset.from_jsonl(settings.test, pokedex, repair=_repair(settings)).contexts
