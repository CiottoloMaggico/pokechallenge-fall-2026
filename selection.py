"""Feature selection on precomputed fold tables, one column at a time.

Each candidate set is compared with the current one fold by fold (paired), with the
corrected repeated k-fold t-test (``stats_tables.corrected_ttest``, Nadeau & Bengio 2003):
the J = folds x repeats differences are not independent, so the naive t is divided by
sqrt(1 + J * n_test / n_train) (2.18 with 5 folds and 3 repeats).

* ``backward`` is the procedure of the multi-seed revalidation (the ``REVALIDATION_DROPS``
  of the feature sets): drop the column whose removal is least harmful, while the one-sided
  corrected p that its removal worsens the log loss is at least ``p_keep`` (0.20).
* ``forward`` is the stepwise search that built ``compact_v4``: add the pool column that lowers
  the log loss the most, if the decrease is significant (corrected p below ``p_add``, 0.05)
  and the accuracy does not drop; otherwise remove the column whose removal keeps the
  accuracy and does not worsen the log loss significantly (corrected t below ``drop_t``,
  1.0); stop when nothing changes or a step would return to an earlier set.

The tables come from ``Gym.fold_tables`` (stateful features fitted inside each training
part), so each comparison only refits the model; ``python main.py select`` builds them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from folds import FoldData
from models import build
from stats_tables import corrected_ttest

PROBA_CLIP = 1e-6

Scores = tuple[np.ndarray, np.ndarray]  # accuracy and log loss per fold


@dataclass
class SelectionReport:
    """The selected columns, their scores per fold and every step taken."""

    columns: list[str]
    scores: Scores
    steps: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready summary: columns, mean accuracy and log loss, steps."""
        return {
            "columns": self.columns,
            "accuracy": float(self.scores[0].mean()),
            "log_loss": float(self.scores[1].mean()),
            "steps": self.steps,
        }


class Selector:
    """Scores column subsets of precomputed fold tables with one registered model."""

    def __init__(self, folds: Sequence[FoldData], model: str) -> None:
        self.folds = folds
        self.model = model
        # n_test / n_train of the corrected t-test
        self.ratio = float(np.mean([len(f[3]) / len(f[1]) for f in folds]))

    @property
    def columns(self) -> list[str]:
        """Every column of the tables."""
        return list(self.folds[0][0].columns)

    def score(self, columns: Sequence[str]) -> Scores:
        """Accuracy and log loss of each fold with only ``columns``."""
        accuracy, loss = [], []
        for x_train, y_train, x_valid, y_valid in self.folds:
            model = build(self.model)
            model.fit(x_train.loc[:, list(columns)], y_train)
            proba = model.predict_proba(x_valid.loc[:, list(columns)])
            proba = np.clip(proba, PROBA_CLIP, 1 - PROBA_CLIP)
            accuracy.append(accuracy_score(y_valid, proba > 0.5))
            loss.append(log_loss(y_valid, proba))
        return np.array(accuracy), np.array(loss)

    def backward(self, start: Sequence[str], p_keep: float = 0.20) -> SelectionReport:
        """Drop the least harmful column while its one-sided corrected p is >= ``p_keep``."""
        report = self._start(start)
        while len(report.columns) > 1:
            best = None
            for column in report.columns:
                trial = self.score([c for c in report.columns if c != column])
                d_acc, d_loss, t, p = compare(trial, report.scores, self.ratio)
                worse_p = p / 2 if t > 0 else 1 - p / 2  # H1: the log loss gets worse
                if worse_p >= p_keep and (best is None or worse_p > best[-1]):
                    best = (column, trial, d_acc, d_loss, t, worse_p)
            if best is None:
                break
            column, trial, d_acc, d_loss, t, worse_p = best
            _step(report, "drop", column, trial, d_acc, d_loss, t, p=worse_p)
        return report

    def forward(
        self,
        start: Sequence[str],
        pool: Sequence[str],
        p_add: float = 0.05,
        drop_t: float = 1.0,
    ) -> SelectionReport:
        """Stepwise search over ``pool``: add a significant column, else drop a useless one."""
        report = self._start(start)
        visited = {frozenset(report.columns)}
        while True:
            action, best = "add", self._best_addition(report, pool, p_add)
            if best is None and len(report.columns) > 1:
                action, best = "drop", self._best_removal(report, drop_t)
            if best is None:
                return report
            column, trial, d_acc, d_loss, t = best
            changed = (
                [*report.columns, column]
                if action == "add"
                else [c for c in report.columns if c != column]
            )
            if frozenset(changed) in visited:  # would go back to an earlier set
                return report
            visited.add(frozenset(changed))
            _step(report, action, column, trial, d_acc, d_loss, t)

    def _start(self, start: Sequence[str]) -> SelectionReport:
        report = SelectionReport(list(start), self.score(start))
        _print("start", "", report)
        return report

    def _best_addition(
        self, report: SelectionReport, pool: Sequence[str], p_add: float
    ) -> tuple[str, Scores, float, float, float] | None:
        best = None
        for column in pool:
            if column in report.columns:
                continue
            trial = self.score([*report.columns, column])
            d_acc, d_loss, t, p = compare(trial, report.scores, self.ratio)
            if d_loss < 0 and p < p_add and d_acc >= 0 and (best is None or d_loss < best[3]):
                best = (column, trial, d_acc, d_loss, t)
        return best

    def _best_removal(
        self, report: SelectionReport, drop_t: float
    ) -> tuple[str, Scores, float, float, float] | None:
        best = None
        for column in report.columns:
            trial = self.score([c for c in report.columns if c != column])
            d_acc, d_loss, t, _ = compare(trial, report.scores, self.ratio)
            if t < drop_t and d_acc >= 0 and (best is None or d_acc > best[2]):
                best = (column, trial, d_acc, d_loss, t)
        return best


def compare(new: Scores, old: Scores, test_train_ratio: float) -> tuple[float, float, float, float]:
    """Mean accuracy change, mean log-loss change, corrected paired t and two-sided p of the
    log loss (``stats_tables.corrected_ttest``); no variance at all counts as no change."""
    d_accuracy, d_loss = new[0] - old[0], new[1] - old[1]
    t, p = corrected_ttest(d_loss, test_train_ratio)
    if np.isnan(t):
        t, p = 0.0, 1.0
    return float(d_accuracy.mean()), float(d_loss.mean()), t, p


def _step(
    report: SelectionReport,
    action: str,
    column: str,
    scores: Scores,
    d_acc: float,
    d_loss: float,
    t: float,
    **extra: float,
) -> None:
    """Apply one step to ``report``, record it and print it."""
    if action == "add":
        report.columns = [*report.columns, column]
    else:
        report.columns = [c for c in report.columns if c != column]
    report.scores = scores
    step = {"action": action, "column": column, "accuracy_change": d_acc}
    report.steps.append({**step, "log_loss_change": d_loss, "t": t, **extra})
    _print(action, column, report)


def _print(action: str, column: str, report: SelectionReport) -> None:
    accuracy, loss = report.scores
    print(
        f"{action:5s} {column:24s} accuracy {accuracy.mean():.4f}  "
        f"log-loss {loss.mean():.4f}  ({len(report.columns)} columns)",
        flush=True,
    )
