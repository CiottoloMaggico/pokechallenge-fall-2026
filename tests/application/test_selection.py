"""Backward and forward selection on small synthetic fold tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from folds import FoldData
from selection import Selector, compare


def folds(n_folds: int = 4, rows: int = 400) -> list[FoldData]:
    """``signal`` predicts the label, ``other`` a little, ``noise_*`` not at all."""
    rng = np.random.default_rng(0)
    x = pd.DataFrame(
        {
            "signal": rng.normal(size=rows),
            "other": rng.normal(size=rows),
            "noise_a": rng.normal(size=rows),
            "noise_b": rng.normal(size=rows),
        }
    )
    y = (2 * x["signal"] + 0.5 * x["other"] + rng.normal(size=rows) > 0).astype(int).to_numpy()
    parts = np.arange(rows) % n_folds
    return [
        (x.loc[parts != k], y[parts != k], x.loc[parts == k], y[parts == k]) for k in range(n_folds)
    ]


def test_backward_drops_the_noise_and_keeps_the_signal() -> None:
    report = Selector(folds(), "logreg").backward(["signal", "noise_a", "noise_b"], p_keep=0.20)
    assert "signal" in report.columns
    assert [step["action"] for step in report.steps] == ["drop"] * len(report.steps)
    assert all(step["p"] >= 0.20 for step in report.steps)


def test_backward_with_p_keep_above_one_drops_nothing() -> None:
    report = Selector(folds(), "logreg").backward(["signal", "noise_a"], p_keep=1.01)
    assert report.columns == ["signal", "noise_a"]
    assert report.steps == []


def test_forward_adds_the_signal_from_the_pool() -> None:
    selector = Selector(folds(), "logreg")
    report = selector.forward(["noise_a"], selector.columns, p_add=0.05, drop_t=1.0)
    assert "signal" in report.columns
    assert report.steps[0] == {**report.steps[0], "action": "add", "column": "signal"}
    summary = report.to_dict()
    assert summary["columns"] == report.columns
    assert summary["log_loss"] == pytest.approx(report.scores[1].mean())


def test_compare_counts_identical_scores_as_no_change() -> None:
    same = (np.array([0.7, 0.8]), np.array([0.5, 0.6]))
    assert compare(same, same, 0.25) == (0.0, 0.0, 0.0, 1.0)
