"""Tables of the gym's results, and the corrected t-test against hand-computed values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stats_tables import (
    calibration_table,
    corrected_ttest,
    paired_table,
    seed_table,
    summary_table,
    threshold_table,
)


def test_the_corrected_test_on_a_hand_computed_case() -> None:
    # mean 2, s^2 = 2.5, J = 5: t = 2 / sqrt((1/5 + 1/4) * 2.5) = 1.8856
    t, p = corrected_ttest([0.0, 1.0, 2.0, 3.0, 4.0], 0.25)
    assert t == pytest.approx(1.885618, abs=1e-6)
    # closed form of df = 4, with tan(a) = t / 2: p = 1 - sin(a) * (1 + cos(a) ** 2 / 2)
    assert p == pytest.approx(0.132419, abs=1e-5)


def test_the_correction_is_smaller_than_the_naive_t() -> None:
    diffs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    naive = diffs.mean() / np.sqrt(diffs.var(ddof=1) / diffs.size)
    assert corrected_ttest(diffs, 0.25)[0] < naive


def test_degenerate_differences() -> None:
    assert corrected_ttest([1.0, 1.0], 0.25) == (np.inf, 0.0)
    assert corrected_ttest([-1.0, -1.0], 0.25) == (-np.inf, 0.0)
    assert corrected_ttest([0.0, 0.0], 0.25) == (0.0, 1.0)
    assert np.isnan(corrected_ttest([1.0], 0.25)[0])


def folds() -> pd.DataFrame:
    """Two models, two seeds, two folds; ``b`` beats ``a`` by 0.1, 0.1, 0.2, 0.2."""
    rows = []
    for seed, base in ((1, 0.6), (2, 0.5)):
        for fold in (0, 1):
            common = {"feature_set": "s", "seed": seed, "fold": fold}
            sizes = {"n_train": 80, "n_valid": 20}
            value = base + fold * 0.1
            gain = 0.1 if seed == 1 else 0.2
            rows.append({**common, **sizes, "model": "a", "accuracy": value})
            rows.append({**common, **sizes, "model": "b", "accuracy": value + gain})
    return pd.DataFrame(rows)


def test_the_summary_separates_folds_from_seeds() -> None:
    a = summary_table(folds(), ["accuracy"]).set_index("model").loc["a"]
    assert a["n_seeds"] == 2
    assert a["accuracy_mean"] == pytest.approx(0.6)
    assert a["accuracy_std"] == pytest.approx(np.std([0.0, 0.1], ddof=1))
    assert a["accuracy_seed_std"] == pytest.approx(np.std([0.65, 0.55], ddof=1))


def test_a_single_seed_has_no_seed_std() -> None:
    table = summary_table(folds().query("seed == 1"), ["accuracy"])
    assert list(table.columns) == ["feature_set", "model", "accuracy_mean", "accuracy_std"]


def test_the_seed_table_has_one_column_per_seed() -> None:
    table = seed_table(folds()).set_index("model")
    assert table.loc["a", "seed 1"] == pytest.approx(0.65)
    assert table.loc["b", "seed 2"] == pytest.approx(0.75)


def test_paired_uses_the_corrected_test_with_the_fold_sizes() -> None:
    table = paired_table(folds(), ("s", "a")).set_index("model")
    b = table.loc["b"]
    assert b["mean_diff"] == pytest.approx(0.15)
    assert b["folds_better"] == 4
    expected = corrected_ttest([0.1, 0.1, 0.2, 0.2], 20 / 80)
    assert (b["t"], b["p"]) == pytest.approx(expected)
    assert table.loc["a", "mean_diff"] == 0


def test_an_unknown_reference_is_refused() -> None:
    with pytest.raises(KeyError):
        paired_table(folds(), ("s", "nope"))


def predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_set": "s",
            "model": "m",
            "proba": [0.1, 0.4, 0.6, 1.0],
            "y_true": [0, 1, 1, 1],
        }
    )


def test_calibration_puts_probability_one_in_the_last_bin() -> None:
    table = calibration_table(predictions(), bins=2)
    assert table["bin"].tolist() == [0, 1]
    assert table["count"].tolist() == [2, 2]
    assert table["observed"].tolist() == [0.5, 1.0]


def test_accuracy_by_threshold() -> None:
    table = threshold_table(predictions(), [0.5, 0.2])
    assert table["accuracy"].tolist() == [0.75, 1.0]
