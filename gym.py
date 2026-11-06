"""Experiment runner: every model x every feature set, on the same folds.

The gym evaluates each combination of models and feature sources with
cross-validation and keeps the *raw* results:

* ``predictions``: one row per (feature set, model, fold, battle) with
  the true label and the predicted probability;
* ``folds``: one row per (feature set, model, fold) with metrics, sizes
  and fit time.

Every plot is derived from these tables by an *extractor*, a pure
function ``GymResult -> pandas.DataFrame`` (see the bottom of this
module). Results can be exported and loaded back, so new plots never
require re-training. Data that needs the fitted model itself (feature
importances, coefficients, ...) is collected during the run by
optional *probes*.

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
* With ``groups`` (e.g. ``Dataset.duplicate_groups()``) identical
  battles always stay in the same fold; every split is checked for
  overlapping rows and groups before training.
* Trained runs save the fitted feature state together with the model.

Example::

    gym = Gym(
        models=[logreg, hgb, EnsembleModel("logreg+hgb", [logreg, hgb])],
        feature_sets=[base_set, full_set],      # FeatureSet("base"), ...
        splitter=RepeatedStratifiedKFold(n_splits=5, n_repeats=2),
    )
    result = gym.evaluate(battles, labels, ids=battle_ids)
    print(result.summary())
    result.export("runs/2026-10-27")
    calibration = result.extract(calibration_table, bins=10)
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from artifacts import FeatureMismatchError, TrainedRun
from interfaces import FeatureSource, FittableFeatureSource, Model, is_stateful

Metric = Callable[[np.ndarray, np.ndarray], float]
# labels, ids and groups may be lists or numpy arrays (e.g. Dataset.labels)
Labels = Sequence[int] | np.ndarray
Ids = Sequence[Any] | np.ndarray
Groups = Sequence[Any] | np.ndarray

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


class Splitter(Protocol):
    """Anything with a scikit-learn style ``split(x, y, groups)``."""

    def split(
        self, x: Any, y: Any, groups: Any, /
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]: ...


class LeakageError(RuntimeError):
    """A split would put the same battle or group in train and validation."""


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


@dataclass(frozen=True)
class FitContext:
    """Everything a probe can look at after one fit."""

    feature_set: str
    model_name: str
    fold: int
    model: Model
    x_valid: pd.DataFrame
    y_valid: np.ndarray


ProbeOutput = Mapping[str, Any] | Sequence[Mapping[str, Any]]
Probe = Callable[[FitContext], ProbeOutput]


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
        """Mean and standard deviation of every metric per combination."""
        return summary_table(self)

    def paired(
        self, reference: tuple[str, str], metric: str = "accuracy"
    ) -> pd.DataFrame:
        """Fold-by-fold difference of every combination vs a reference."""
        return paired_table(self, reference, metric)

    def extract(
        self, extractor: Callable[..., pd.DataFrame], **kwargs: Any
    ) -> pd.DataFrame:
        """Apply a plot-specific extractor to these results."""
        return extractor(self, **kwargs)

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


class Gym:
    """Runs every model on every feature source with shared folds."""

    def __init__(
        self,
        models: Model | Sequence[Model],
        feature_sets: FeatureSource | Sequence[FeatureSource],
        splitter: Splitter | None = None,
        metrics: Mapping[str, Metric] | None = None,
        probes: Mapping[str, Probe] | None = None,
    ) -> None:
        self._models: list[Model] = (
            [models]
            if isinstance(models, Model)
            else list(cast("Sequence[Model]", models))
        )
        sources: list[FeatureSource] = (
            [feature_sets]
            if isinstance(feature_sets, FeatureSource)
            else list(cast("Sequence[FeatureSource]", feature_sets))
        )
        self._check_unique("feature set", [s.name for s in sources])
        self._feature_sets = {s.name: s for s in sources}
        self._splitter = splitter
        self._metrics = dict(metrics or DEFAULT_METRICS)
        self._probes = dict(probes or {})
        self._validate()

    def evaluate(
        self,
        battles: Sequence[Any],
        labels: Labels,
        ids: Ids | None = None,
        groups: Groups | None = None,
        unlabeled: Sequence[Any] = (),
    ) -> GymResult:
        """Cross-validate every (feature set, model) combination.

        ``groups`` keeps battles with the same group in the same fold
        (use ``Dataset.duplicate_groups()``). ``unlabeled`` battles (e.g.
        the test set) may be used by label-free stateful features only.
        """
        y = np.asarray(labels)
        row_ids = np.asarray(ids) if ids is not None else np.arange(len(y))
        group_ids = None if groups is None else np.asarray(groups)
        sizes = {len(y), len(battles), len(row_ids)}
        if group_ids is not None:
            sizes.add(len(group_ids))
        if len(sizes) != 1:
            raise ValueError("battles, labels, ids and groups must have equal length")

        splits = self._splits(y, group_ids)
        predictions: list[pd.DataFrame] = []
        folds: list[dict[str, Any]] = []
        probe_rows: dict[str, list[dict[str, Any]]] = {
            name: [] for name in self._probes
        }
        for fs_name, source in self._feature_sets.items():
            shared = None if is_stateful(source) else source.to_frame(battles)
            for fold, (train, valid) in enumerate(splits):
                if shared is None:
                    x_train, x_valid = _fold_features(
                        cast(FittableFeatureSource, source),
                        battles,
                        y,
                        (train, valid),
                        unlabeled,
                        group_ids,
                    )
                else:
                    x_train, x_valid = shared.iloc[train], shared.iloc[valid]
                for model in self._models:
                    tags = {
                        "feature_set": fs_name,
                        "model": model.name,
                        "fold": fold,
                    }
                    preds, fold_row = self._run_one(
                        model,
                        (x_train, y[train]),
                        (x_valid, y[valid]),
                        valid,
                        row_ids,
                        tags,
                        probe_rows,
                    )
                    predictions.append(preds)
                    folds.append(fold_row)

        return GymResult(
            predictions=pd.concat(predictions, ignore_index=True),
            folds=pd.DataFrame(folds),
            probes={k: pd.DataFrame(v) for k, v in probe_rows.items()},
            meta=self._meta(len(y), len(splits), group_ids, len(unlabeled)),
        )

    def train(
        self,
        battles: Sequence[Any],
        labels: Labels,
        unlabeled: Sequence[Any] = (),
        groups: Groups | None = None,
        repair: Mapping[str, Any] | None = None,
    ) -> list[TrainedRun]:
        """Fit every combination on all training data (before a submission).

        Stateful feature sets are cloned and fitted on ``battles`` (plus
        ``unlabeled`` for label-free features); the fitted clone travels
        with the model inside the returned :class:`TrainedRun`. The model
        is trained on out-of-fold values of label-aware features, grouped
        by ``groups`` (e.g. ``Dataset.duplicate_groups()``). ``repair`` is the
        manifest of the repair applied to ``battles``, saved with each run.
        """
        y = np.asarray(labels)
        runs: list[TrainedRun] = []
        for source in self._feature_sets.values():
            fitted_source: FeatureSource = source
            if is_stateful(source):
                fittable = cast(FittableFeatureSource, source).clone()
                group_list = None if groups is None else np.asarray(groups).tolist()
                x = fittable.fit_transform(battles, y.tolist(), unlabeled, group_list)
                fitted_source = fittable
            else:
                x = source.to_frame(battles)
            columns = tuple(str(c) for c in x.columns)
            for model in self._models:
                fitted = model.clone().fit(x, y)
                runs.append(TrainedRun(fitted_source, fitted, columns, repair))
        return runs

    def _splits(
        self, y: np.ndarray, groups: np.ndarray | None
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Folds from the splitter, checked for overlapping rows or groups."""
        return make_splits(self._splitter, y, groups)

    def _run_one(
        self,
        model: Model,
        train_data: tuple[pd.DataFrame, np.ndarray],
        valid_data: tuple[pd.DataFrame, np.ndarray],
        valid: np.ndarray,
        row_ids: np.ndarray,
        tags: dict[str, Any],
        probe_rows: dict[str, list[dict[str, Any]]],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fit on the training part, predict validation, score, run probes."""
        x_train, y_train = train_data
        x_valid, y_valid = valid_data
        fitted, seconds = _timed_fit(model, x_train, y_train)
        proba = fitted.predict_proba(x_valid)
        preds = pd.DataFrame(
            {
                **tags,
                "row": valid,
                "id": row_ids[valid],
                "y_true": y_valid,
                "proba": proba,
            }
        )
        fold_row = {
            **tags,
            "n_train": len(y_train),
            "n_valid": len(y_valid),
            "n_features": x_valid.shape[1],
            "fit_seconds": seconds,
            **self._score(y_valid, proba),
        }
        context = FitContext(
            tags["feature_set"],
            model.name,
            tags["fold"],
            fitted,
            x_valid,
            y_valid,
        )
        self._run_probes(context, tags, probe_rows)
        return preds, fold_row

    def _validate(self) -> None:
        if not self._models:
            raise ValueError("the gym needs at least one model")
        if not self._feature_sets:
            raise ValueError("the gym needs at least one feature set")
        self._check_unique("model", [m.name for m in self._models])

    @staticmethod
    def _check_unique(kind: str, names: Sequence[str]) -> None:
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"{kind} names must be unique: {duplicates}")

    def _score(self, y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
        return {name: float(fn(y, proba)) for name, fn in self._metrics.items()}

    def _run_probes(
        self,
        context: FitContext,
        tags: dict[str, Any],
        rows: dict[str, list[dict[str, Any]]],
    ) -> None:
        for name, probe in self._probes.items():
            output = probe(context)
            records = [output] if isinstance(output, Mapping) else list(output)
            rows[name].extend({**tags, **dict(r)} for r in records)

    def _meta(
        self,
        n_rows: int,
        n_folds: int,
        groups: np.ndarray | None,
        n_unlabeled: int,
    ) -> dict[str, Any]:
        return {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_rows": n_rows,
            "n_folds": n_folds,
            "splitter": repr(self._splitter) if self._splitter else "default",
            "grouped": groups is not None,
            "n_groups": None if groups is None else int(np.unique(groups).size),
            "n_unlabeled": n_unlabeled,
            "refit_per_fold": [
                name for name, s in self._feature_sets.items() if is_stateful(s)
            ],
            "metrics": list(self._metrics),
            "models": {m.name: _jsonable(m.get_params()) for m in self._models},
            "feature_sets": list(self._feature_sets),
            "probes": list(self._probes),
        }


def make_splits(
    splitter: Splitter | None, y: np.ndarray, groups: np.ndarray | None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds from ``splitter`` (grouped default), checked for leakage."""
    chosen: Splitter = splitter or (
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        if groups is not None
        else StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    splits = [
        (np.asarray(tr), np.asarray(va))
        for tr, va in chosen.split(np.zeros(len(y)), y, groups)
    ]
    for fold, (train, valid) in enumerate(splits):
        if np.intersect1d(train, valid).size:
            raise LeakageError(f"fold {fold}: rows in both train and validation")
        if groups is not None:
            shared = np.intersect1d(groups[train], groups[valid])
            if shared.size:
                raise LeakageError(
                    f"fold {fold}: groups {shared[:5].tolist()} in both train "
                    "and validation (use a group-aware splitter)"
                )
    return splits


FoldData = tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]


def prepare_folds(
    source: FeatureSource,
    battles: Sequence[Any],
    labels: Labels,
    splitter: Splitter | None = None,
    groups: Groups | None = None,
    unlabeled: Sequence[Any] = (),
    progress: Callable[[int, int, float], None] | None = None,
) -> list[FoldData]:
    """Feature tables of every fold, built with the gym's leakage rules.

    Returns ``(x_train, y_train, x_valid, y_valid)`` per fold. Computing
    them once lets many models (e.g. a hyperparameter search) reuse
    them: stateful features are still fitted on each training part only.
    ``progress(done, total, seconds)`` is called after each fold.
    """
    y = np.asarray(labels)
    group_ids = None if groups is None else np.asarray(groups)
    splits = make_splits(splitter, y, group_ids)
    shared = None if is_stateful(source) else source.to_frame(battles)
    folds: list[FoldData] = []
    start = time.perf_counter()
    for train, valid in splits:
        if shared is None:
            x_train, x_valid = _fold_features(
                cast(FittableFeatureSource, source),
                battles,
                y,
                (train, valid),
                unlabeled,
                group_ids,
            )
        else:
            x_train, x_valid = shared.iloc[train], shared.iloc[valid]
        folds.append((x_train, y[train], x_valid, y[valid]))
        if progress is not None:
            progress(len(folds), len(splits), time.perf_counter() - start)
    return folds


def _fold_features(
    source: FittableFeatureSource,
    battles: Sequence[Any],
    y: np.ndarray,
    split: tuple[np.ndarray, np.ndarray],
    unlabeled: Sequence[Any],
    groups: np.ndarray | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a fresh clone on the training battles only, then transform both.

    Training rows of label-aware features are out of fold (inner folds
    grouped like the outer ones), validation rows use the full fit.
    """
    train, valid = split
    train_battles = [battles[i] for i in train]
    valid_battles = [battles[i] for i in valid]
    fold_source = source.clone()
    x_train = fold_source.fit_transform(
        train_battles,
        y[train].tolist(),
        unlabeled,
        None if groups is None else groups[train].tolist(),
    )
    x_valid = fold_source.to_frame(valid_battles)
    if list(x_train.columns) != list(x_valid.columns):
        raise FeatureMismatchError(
            f"feature set {source.name!r}: train and validation columns differ"
        )
    return x_train, x_valid


def _timed_fit(model: Model, x: pd.DataFrame, y: np.ndarray) -> tuple[Model, float]:
    start = time.perf_counter()
    fitted = model.clone().fit(x, y)
    return fitted, time.perf_counter() - start


def _jsonable(params: Mapping[str, Any]) -> dict[str, Any]:
    simple = (str, int, float, bool, type(None))
    return {k: v if isinstance(v, simple) else repr(v) for k, v in params.items()}


def _mask(table: pd.DataFrame, feature_set: str, model: str) -> np.ndarray:
    """Boolean row mask for one (feature set, model) combination."""
    fs_match = np.asarray(table["feature_set"]) == feature_set
    model_match = np.asarray(table["model"]) == model
    return fs_match & model_match


SEED_STD_METRICS = ("accuracy", "log_loss")  # metrics that also get a seed_std


def summary_table(result: GymResult) -> pd.DataFrame:
    """Mean and std of each metric per (feature set, model).

    With one seed (no ``seed`` column, or a single value) the columns are
    ``<metric>_mean`` and ``<metric>_std`` over the folds. With several
    seeds of the fold assignment (``main.py evaluate --n-seeds``):

    * ``<metric>_mean``: mean over every seed x fold;
    * ``<metric>_std``: fold-to-fold std within a seed, averaged over the
      seeds (same meaning as with one seed);
    * ``<metric>_seed_std``: std across seeds of the per-seed means, i.e.
      how much the number moves with the luck of the fold assignment
      (accuracy and log loss only, to keep the table short).
    """
    folds = result.folds
    metrics = result.metric_names
    if "seed" not in folds or folds["seed"].nunique() < 2:
        aggregations = {
            f"{metric}_{stat}": (metric, stat)
            for metric in metrics
            for stat in ("mean", "std")
        }
        grouped = folds.groupby(["feature_set", "model"])
        return grouped.agg(**aggregations).reset_index()
    by_seed = folds.groupby(["feature_set", "model", "seed"])[metrics]
    means, stds = by_seed.mean(), by_seed.std()
    keys = ["feature_set", "model"]
    columns: dict[str, pd.Series] = {}
    for metric in metrics:
        columns[f"{metric}_mean"] = means[metric].groupby(level=keys).mean()
        columns[f"{metric}_std"] = stds[metric].groupby(level=keys).mean()
        if metric in SEED_STD_METRICS:
            columns[f"{metric}_seed_std"] = means[metric].groupby(level=keys).std()
    table = pd.DataFrame(columns).reset_index()
    table.insert(2, "n_seeds", means.groupby(level=keys).size().to_numpy())
    return table


def seed_table(result: GymResult, metric: str = "accuracy") -> pd.DataFrame:
    """Mean ``metric`` over the folds of each seed: one row per (feature
    set, model), one column per seed."""
    folds = result.folds
    if "seed" not in folds:
        raise KeyError("these results have a single seed")
    table = folds.pivot_table(
        index=["feature_set", "model"], columns="seed", values=metric, aggfunc="mean"
    )
    table.columns = [f"seed {s}" for s in table.columns]
    return table.reset_index()


def corrected_ttest(
    diffs: Sequence[float] | np.ndarray, test_train_ratio: float
) -> tuple[float, float]:
    """Corrected repeated k-fold t-test (Nadeau & Bengio 2003; Bouckaert &
    Frank 2004) of paired per-fold differences.

    The J differences of r repetitions of k-fold CV are not independent
    (training sets overlap and the repetitions reuse the same battles), so
    the variance of their mean is ``(1/J + n_test/n_train) * s^2`` instead
    of ``s^2 / J``: the naive t is divided by ``sqrt(1 + J * n_test/n_train)``.
    Note the floor: more seeds shrink 1/J, never the n_test/n_train term.
    Returns ``(t, two-sided p)`` with J - 1 degrees of freedom.
    """
    d = np.asarray(diffs, dtype=float)
    j = d.size
    if j < 2:
        return float("nan"), float("nan")
    var = d.var(ddof=1)
    if var == 0:
        return (0.0, 1.0) if d.mean() == 0 else (float(np.sign(d.mean()) * np.inf), 0.0)
    t = d.mean() / np.sqrt((1 / j + test_train_ratio) * var)
    return float(t), float(2 * stats.t.sf(abs(t), j - 1))


def paired_table(
    result: GymResult, reference: tuple[str, str], metric: str = "accuracy"
) -> pd.DataFrame:
    """Per-fold ``metric`` difference of each combination vs reference.

    Columns: feature_set, model, mean_diff, std_diff, folds_better (seed x
    fold pairs where the combination is better), and ``t`` / ``p`` of the
    corrected repeated k-fold t-test (:func:`corrected_ttest`), which does
    not treat overlapping folds and repeated seeds as independent.
    """
    folds = result.folds
    is_ref = _mask(folds, reference[0], reference[1])
    if not bool(is_ref.any()):
        raise KeyError(f"reference {reference} not in the results")
    keys = ["seed", "fold"] if "seed" in folds else ["fold"]
    ref = folds.loc[is_ref, [*keys, metric]].rename(columns={metric: "reference"})
    merged = folds.merge(ref, on=keys)
    merged["diff"] = merged[metric] - merged["reference"]
    merged["better"] = (merged["diff"] > 0).astype(int)
    ratio = float((folds["n_valid"] / folds["n_train"]).mean())
    table = merged.groupby(["feature_set", "model"]).agg(
        mean_diff=("diff", "mean"),
        std_diff=("diff", "std"),
        folds_better=("better", "sum"),
    )
    tests = merged.groupby(["feature_set", "model"])["diff"].apply(
        lambda d: corrected_ttest(d.to_numpy(), ratio)
    )
    table["t"] = [t for t, _ in tests.loc[table.index]]
    table["p"] = [p for _, p in tests.loc[table.index]]
    return table.reset_index().sort_values("mean_diff", ascending=False)


def calibration_table(result: GymResult, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed win rate per probability bin."""
    preds = result.predictions.copy()
    preds["bin"] = np.minimum((preds["proba"] * bins).astype(int), bins - 1)
    grouped = preds.groupby(["feature_set", "model", "bin"])
    table = grouped.agg(
        mean_proba=("proba", "mean"),
        observed=("y_true", "mean"),
        count=("y_true", "size"),
    )
    return table.reset_index()


def threshold_table(
    result: GymResult, thresholds: Sequence[float] | None = None
) -> pd.DataFrame:
    """Accuracy as a function of the decision threshold."""
    grid = np.linspace(0.3, 0.7, 41) if thresholds is None else thresholds
    preds = result.predictions
    combos = preds[["feature_set", "model"]].drop_duplicates()
    rows: list[dict[str, Any]] = []
    for fs, model in combos.itertuples(index=False, name=None):
        group = preds[_mask(preds, fs, model)]
        y, p = np.asarray(group["y_true"]), np.asarray(group["proba"])
        for t in grid:
            rows.append(
                {
                    "feature_set": fs,
                    "model": model,
                    "threshold": float(t),
                    "accuracy": float(((p > t).astype(int) == y).mean()),
                }
            )
    return pd.DataFrame(rows)


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
