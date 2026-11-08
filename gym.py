"""Experiment runner: every model x every feature set, on the same folds.

The gym evaluates each combination of models and feature sources with
cross-validation and keeps the *raw* results:

* ``predictions``: one row per (feature set, model, fold, seed, battle) with
  the true label and the predicted probability;
* ``folds``: one row per (feature set, model, fold, seed) with metrics, sizes
  and fit time.

Every table for a plot is derived from these (``stats_tables.py``). Results can
be exported and loaded back, so new plots never require re-training. Data that
needs the fitted model itself (feature importances, coefficients, ...) is
collected during the run by optional *probes*.

All combinations share the same folds, so differences between models
or feature sets can be compared fold by fold (paired comparison).

Leakage protection
------------------
* Models are cloned and fitted on the training part of each fold only.
* Stateful feature sets (features with ``fit``) are cloned and fitted
  inside every fold on the training battles (and their labels, for
  features that use them); validation battles never shape their state.
* Label-aware features (e.g. target encoding) give the model
  out-of-fold values on the training rows, never values fitted on the
  row's own label.
  Optional ``unlabeled`` battles (e.g. the test set) may feed label-free
  features only.
* Rows of the same group (``Dataset.duplicate_groups()``) always stay in
  the same fold; every split is checked for overlapping rows and groups
  before training.
* Trained runs save the fitted feature state together with the model.

Example::

    gym = Gym(
        models=[logreg, hgb, EnsembleModel("logreg+hgb", [logreg, hgb])],
        feature_sets=[base_set, full_set],      # FeatureSet("base"), ...
        plan=FoldPlan(n_folds=5, seeds=(33, 66, 133)),
    )
    result = gym.evaluate(train.labelled())
    print(result.summary())
    result.export("results/evaluations/2026-10-27")
"""

from __future__ import annotations

import json
import multiprocessing as mp
import pickle
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from artifacts import TrainedRun
from folds import FoldData, FoldPlan, fold_frames, make_splits, shared_table, table_path
from interfaces import (
    FeatureSource,
    FitContext,
    FittableFeatureSource,
    LabelledBattles,
    Model,
    Probe,
    Splitter,
    is_stateful,
)
from stats_tables import paired_table, seed_table, summary_table

Metric = Callable[[np.ndarray, np.ndarray], float]
Progress = Callable[[int, int, float], None]

DEFAULT_METRICS: dict[str, Metric] = {
    "accuracy": lambda y, p: float(accuracy_score(y, (p > 0.5).astype(int))),
    "log_loss": lambda y, p: float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))),
    "roc_auc": lambda y, p: float(roc_auc_score(y, p)),
    "brier": lambda y, p: float(brier_score_loss(y, p)),
}

PREDICTIONS_FILE = "predictions.csv"
FOLDS_FILE = "folds.csv"
META_FILE = "meta.json"
PROBES_DIR = "probes"


@dataclass(frozen=True)
class PrecomputedFeatures:
    """Feature source backed by an already computed table.

    With ``key`` (a function battle -> id) the table must be indexed by
    that id, and any subset of battles can be looked up. Without it,
    rows must be aligned one to one with the battles given.

    The gym treats it as stateless: the table must have been computed
    battle by battle, without labels and without statistics learned on
    other battles, or the evaluation will leak.
    """

    name: str
    table: pd.DataFrame
    key: Callable[[Any], Any] | None = None

    def to_frame(self, battles: Iterable[Any]) -> pd.DataFrame:
        battles = list(battles)
        if self.key is not None:
            ids = [self.key(b) for b in battles]
            missing = [i for i in ids if i not in self.table.index]
            if missing:
                raise KeyError(f"ids not in the table: {missing[:5]}")
            return self.table.reindex(ids).reset_index(drop=True)
        if len(battles) != len(self.table):
            raise ValueError(
                f"precomputed table has {len(self.table)} rows, "
                f"got {len(battles)} battles: pass key= to look rows up"
            )
        return self.table.reset_index(drop=True)


@dataclass
class GymResult:
    """Raw results of an evaluation, plus helpers to analyse them."""

    predictions: pd.DataFrame
    folds: pd.DataFrame
    probes: dict[str, pd.DataFrame] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def metric_names(self) -> list[str]:
        return list(self.meta.get("metrics", []))

    def summary(self) -> pd.DataFrame:
        """Mean, fold std and seed std of every metric per combination."""
        return summary_table(self.folds, self.metric_names)

    def by_seed(self, metric: str = "accuracy") -> pd.DataFrame:
        """Mean ``metric`` of each seed: one row per combination, one column per seed."""
        return seed_table(self.folds, metric)

    def paired(self, reference: tuple[str, str], metric: str = "accuracy") -> pd.DataFrame:
        """Fold-by-fold difference of every combination vs a reference (corrected t-test)."""
        return paired_table(self.folds, reference, metric)

    def export(self, directory: str | Path) -> Path:
        """Write all raw tables to ``directory`` (created if needed)."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.predictions.to_csv(out / PREDICTIONS_FILE, index=False)
        self.folds.to_csv(out / FOLDS_FILE, index=False)
        (out / META_FILE).write_text(json.dumps(self.meta, indent=2))
        if self.probes:
            (out / PROBES_DIR).mkdir(exist_ok=True)
            for name, table in self.probes.items():
                table.to_csv(out / PROBES_DIR / f"{name}.csv", index=False)
        return out

    @classmethod
    def combine(cls, results: Mapping[int, GymResult]) -> GymResult:
        """One result from evaluations of the same combinations on folds
        drawn with different seeds (``{seed: result}``): every table gets a
        ``seed`` column, the meta lists the seeds and their splitters."""
        if not results:
            raise ValueError("no results to combine")
        seeds = list(results)

        def stack(tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
            return pd.concat(list(tables), ignore_index=True)

        first = results[seeds[0]]
        meta = dict(first.meta)
        meta["seeds"] = seeds
        meta["splitter"] = {str(s): r.meta.get("splitter") for s, r in results.items()}
        return cls(
            predictions=stack(r.predictions.assign(seed=s) for s, r in results.items()),
            folds=stack(r.folds.assign(seed=s) for s, r in results.items()),
            probes={
                name: stack(r.probes[name].assign(seed=s) for s, r in results.items())
                for name in first.probes
            },
            meta=meta,
        )

    @classmethod
    def load(cls, directory: str | Path) -> GymResult:
        """Load results written by :meth:`export`."""
        src = Path(directory)
        probes_dir = src / PROBES_DIR
        probes = (
            {p.stem: pd.read_csv(p) for p in sorted(probes_dir.glob("*.csv"))}
            if probes_dir.is_dir()
            else {}
        )
        return cls(
            predictions=pd.read_csv(src / PREDICTIONS_FILE),
            folds=pd.read_csv(src / FOLDS_FILE),
            probes=probes,
            meta=json.loads((src / META_FILE).read_text()),
        )


_SHARED: dict[str, tuple[Gym, LabelledBattles, Sequence[Any]]] = {}


def _evaluate_shared(seed: int, splitter: Splitter) -> GymResult:
    gym, data, unlabeled = _SHARED["job"]
    return gym._evaluate_seed(seed, splitter, data, unlabeled)


class Gym:
    """Runs every model on every feature source with shared folds.

    ``jobs`` evaluates that many seeds in parallel with fork: never pass more than 1 for
    models or features that use LightGBM, XGBoost or CatBoost, whose OpenMP hangs in forked
    workers (``registry.fork_unsafe_libraries``).
    """

    def __init__(
        self,
        models: Sequence[Model],
        feature_sets: Sequence[FeatureSource],
        plan: FoldPlan = FoldPlan(),  # noqa: B008  (immutable)
        metrics: Mapping[str, Metric] = DEFAULT_METRICS,
        probes: Mapping[str, Probe] | None = None,
        jobs: int = 1,
    ) -> None:
        _check_unique("feature set", [s.name for s in feature_sets])
        _check_unique("model", [m.name for m in models])
        if not models:
            raise ValueError("the gym needs at least one model")
        if not feature_sets:
            raise ValueError("the gym needs at least one feature set")
        self._models = list(models)
        self._feature_sets = {s.name: s for s in feature_sets}
        self._plan = plan
        self._metrics = dict(metrics)
        self._probes = dict(probes or {})
        self._jobs = jobs

    def evaluate(self, data: LabelledBattles, unlabeled: Sequence[Any] = ()) -> GymResult:
        """Cross-validate every (feature set, model) combination on each seed of the plan.

        ``unlabeled`` battles (e.g. the test set) may be used by label-free stateful
        features only.
        """
        splitters = list(self._plan.splitters())
        if self._jobs > 1 and len(splitters) > 1 and "fork" in mp.get_all_start_methods():
            # forked workers share the replayed battles (copy on write) instead
            # of receiving a pickled copy each, which would multiply the memory
            _SHARED["job"] = (self, data, unlabeled)
            with mp.get_context("fork").Pool(min(self._jobs, len(splitters))) as pool:
                runs = pool.starmap(_evaluate_shared, splitters, chunksize=1)
        else:
            runs = [self._evaluate_seed(s, sp, data, unlabeled) for s, sp in splitters]
        return GymResult.combine(dict(zip(self._plan.seeds, runs, strict=True)))

    def train(
        self,
        data: LabelledBattles,
        unlabeled: Sequence[Any] = (),
        repair: Mapping[str, Any] | None = None,
    ) -> list[TrainedRun]:
        """Fit every combination on all training data (before a submission).

        Stateful feature sets are cloned and fitted on the rows (plus ``unlabeled`` for
        label-free features); the fitted clone travels with the model inside the returned
        :class:`TrainedRun`. The model is trained on out-of-fold values of label-aware
        features, grouped like the rows. ``repair`` is the manifest of the repair applied to
        the battles, saved with each run.
        """
        runs: list[TrainedRun] = []
        for source in self._feature_sets.values():
            fitted_source: FeatureSource = source
            if is_stateful(source):
                fittable = cast(FittableFeatureSource, source).clone()
                x = fittable.fit_transform(
                    data.items, data.labels.tolist(), unlabeled, data.groups.tolist()
                )
                fitted_source = fittable
            else:
                x = source.to_frame(data.items)
            columns = tuple(str(c) for c in x.columns)
            for model in self._models:
                fitted = model.clone().fit(x, data.labels)
                runs.append(TrainedRun(fitted_source, fitted, columns, repair))
        return runs

    @staticmethod
    def fold_tables(
        data: LabelledBattles,
        feature_set: FeatureSource,
        plan: FoldPlan,
        unlabeled: Sequence[Any] = (),
        cache: Path | None = None,
        progress: Progress | None = None,
    ) -> dict[int, list[FoldData]]:
        """``(x_train, y_train, x_valid, y_valid)`` of every fold, by seed.

        Built with the gym's leakage rules, so many models (a hyperparameter search) can reuse
        them. With ``cache`` (a directory) the tables are also written for
        :func:`folds.read_fold_tables`. ``progress(done, total, seconds)`` follows each fold.
        """
        shared = shared_table(feature_set, data)
        total = plan.n_folds * len(plan.seeds)
        tables: dict[int, list[FoldData]] = {}
        done, start = 0, time.perf_counter()
        for seed, splitter in plan.splitters():
            folds: list[FoldData] = []
            for split in make_splits(splitter, data.labels, data.groups):
                x_train, x_valid = fold_frames(feature_set, shared, data, split, unlabeled)
                folds.append((x_train, data.labels[split[0]], x_valid, data.labels[split[1]]))
                done += 1
                if progress is not None:
                    progress(done, total, time.perf_counter() - start)
            tables[seed] = folds
            if cache is not None:
                path = table_path(cache, feature_set.name, plan.n_folds, seed, bool(unlabeled))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(pickle.dumps(folds))
        return tables

    def _evaluate_seed(
        self, seed: int, splitter: Splitter, data: LabelledBattles, unlabeled: Sequence[Any]
    ) -> GymResult:
        """One full cross-validation with the folds of ``splitter``."""
        started = time.perf_counter()
        splits = make_splits(splitter, data.labels, data.groups)
        predictions: list[pd.DataFrame] = []
        folds: list[dict[str, Any]] = []
        probe_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in self._probes}
        for fs_name, source in self._feature_sets.items():
            shared = shared_table(source, data)
            for fold, split in enumerate(splits):
                x_train, x_valid = fold_frames(source, shared, data, split, unlabeled)
                for model in self._models:
                    tags = {"feature_set": fs_name, "model": model.name, "fold": fold}
                    preds, fold_row = self._run_one(
                        model, x_train, x_valid, data, split, tags, probe_rows
                    )
                    predictions.append(preds)
                    folds.append(fold_row)
        print(f"  seed {seed} done ({time.perf_counter() - started:.0f} s)", flush=True)
        return GymResult(
            predictions=pd.concat(predictions, ignore_index=True),
            folds=pd.DataFrame(folds),
            probes={k: pd.DataFrame(v) for k, v in probe_rows.items()},
            meta=self._meta(data, splitter, len(splits), len(unlabeled)),
        )

    def _run_one(
        self,
        model: Model,
        x_train: pd.DataFrame,
        x_valid: pd.DataFrame,
        data: LabelledBattles,
        split: tuple[np.ndarray, np.ndarray],
        tags: dict[str, Any],
        probe_rows: dict[str, list[dict[str, Any]]],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fit on the training part, predict validation, score, run probes."""
        train, valid = split
        y_valid = data.labels[valid]
        start = time.perf_counter()
        fitted = model.clone().fit(x_train, data.labels[train])
        seconds = time.perf_counter() - start
        proba = fitted.predict_proba(x_valid)
        preds = pd.DataFrame(
            {**tags, "row": valid, "id": data.ids[valid], "y_true": y_valid, "proba": proba}
        )
        fold_row = {
            **tags,
            "n_train": len(train),
            "n_valid": len(valid),
            "n_features": x_valid.shape[1],
            "fit_seconds": seconds,
            **{name: float(fn(y_valid, proba)) for name, fn in self._metrics.items()},
        }
        context = FitContext(
            tags["feature_set"], model.name, tags["fold"], fitted, x_valid, y_valid
        )
        for name, probe in self._probes.items():
            output = probe(context)
            records = [output] if isinstance(output, Mapping) else list(output)
            probe_rows[name].extend({**tags, **dict(r)} for r in records)
        return preds, fold_row

    def _meta(
        self, data: LabelledBattles, splitter: Splitter, n_folds: int, n_unlabeled: int
    ) -> dict[str, Any]:
        return {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_rows": len(data.labels),
            "n_folds": n_folds,
            "splitter": repr(splitter),
            "grouped": True,
            "n_groups": int(np.unique(data.groups).size),
            "n_unlabeled": n_unlabeled,
            "refit_per_fold": [name for name, s in self._feature_sets.items() if is_stateful(s)],
            "metrics": list(self._metrics),
            "models": {m.name: _jsonable(m.get_params()) for m in self._models},
            "feature_sets": list(self._feature_sets),
            "probes": list(self._probes),
        }


def _check_unique(kind: str, names: Sequence[str]) -> None:
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"{kind} names must be unique: {duplicates}")


def _jsonable(params: Mapping[str, Any]) -> dict[str, Any]:
    simple = (str, int, float, bool, type(None))
    return {k: v if isinstance(v, simple) else repr(v) for k, v in params.items()}


def permutation_importance_probe(n_repeats: int = 3, seed: int = 0) -> Probe:
    """Accuracy drop when each feature is shuffled in the valid fold."""

    def probe(ctx: FitContext) -> list[dict[str, Any]]:
        rng = np.random.default_rng(seed)
        y = ctx.y_valid
        base = float((ctx.model.predict(ctx.x_valid) == y).mean())
        rows = []
        for column in ctx.x_valid.columns:
            drops = []
            for _ in range(n_repeats):
                shuffled = ctx.x_valid.copy()
                values = np.asarray(shuffled[column])
                shuffled[column] = rng.permutation(values)
                acc = float((ctx.model.predict(shuffled) == y).mean())
                drops.append(base - acc)
            rows.append({"feature": column, "accuracy_drop": float(np.mean(drops))})
        return rows

    return probe
