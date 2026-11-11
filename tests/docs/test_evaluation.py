"""The scripts of ``docs/evaluation``: the blend, the cut of the battles, the nested subsets of
the learning curve and the ROC points of the figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "evaluation"))

from blend import blend, blend_files  # noqa: E402  # pyright: ignore[reportMissingImports]
from learning_curve import nested_subset  # noqa: E402  # pyright: ignore[reportMissingImports]
from truncate import truncate  # noqa: E402  # pyright: ignore[reportMissingImports]


def test_the_blend_averages_the_logits_and_clips_certain_probabilities() -> None:
    a, b = np.array([0.9, 0.2, 1.0]), np.array([0.6, 0.4, 0.0])
    found = blend(a, b)
    assert found[:2] == pytest.approx(expit((logit(a[:2]) + logit(b[:2])) / 2))
    assert found[2] == pytest.approx(0.5)  # 1 and 0 clipped symmetrically
    assert blend(a, b, 1.0)[:2] == pytest.approx(a[:2])


def test_the_blend_of_two_files_is_a_submission(tmp_path: Path) -> None:
    ids = [7, 3, 5]
    for name, p in (("a", [0.9, 0.4, 0.2]), ("b", [0.8, 0.7, 0.1])):
        table = pd.DataFrame({"battle_id": ids, "player_won": 0, "p_player_won": p})
        table.to_csv(tmp_path / f"{name}.csv", index=False)
    blend_files(tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "s.csv", tmp_path / "p.csv")
    found = pd.read_csv(tmp_path / "s.csv")
    assert list(found.columns) == ["battle_id", "player_won"]
    assert found["battle_id"].tolist() == ids and found["player_won"].tolist() == [1, 1, 0]
    assert pd.read_csv(tmp_path / "p.csv")["p_player_won"].between(0, 1).all()
    pd.DataFrame({"battle_id": [1, 2, 3], "p_player_won": 0.5}).to_csv(tmp_path / "c.csv")
    with pytest.raises(SystemExit, match="same battles"):
        blend_files(tmp_path / "a.csv", tmp_path / "c.csv", tmp_path / "s.csv")


def test_truncate_keeps_the_first_turns_and_everything_else(tmp_path: Path) -> None:
    records = [{"battle_id": k, "battle_timeline": list(range(n)), "player_won": 1}
               for k, n in enumerate((30, 3))]  # fmt: skip
    source = tmp_path / "train.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in records))
    truncate(source, 5, tmp_path / "cut.jsonl")
    found = [json.loads(line) for line in (tmp_path / "cut.jsonl").read_text().splitlines()]
    assert [r["battle_timeline"] for r in found] == [[0, 1, 2, 3, 4], [0, 1, 2]]
    assert [(r["battle_id"], r["player_won"]) for r in found] == [(0, 1), (1, 1)]


def test_the_subsets_of_the_learning_curve_are_nested_and_keep_duplicates_together() -> None:
    groups = np.array([0, 1, 1, 2, 3, 4, 4, 4, 5, 6, 7, 8])
    train = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 11])  # rows 4 and 10 are validation
    small, large = nested_subset(train, groups, 3), nested_subset(train, groups, 6)
    assert set(small) < set(large) <= set(train)
    assert 3 <= len(small) <= 5 and 6 <= len(large) <= 8
    for subset in (small, large):
        inside = set(groups[subset])
        assert not [row for row in train if groups[row] in inside and row not in subset]
    assert sorted(nested_subset(train, groups, 99)) == sorted(train)


def test_the_roc_points_of_the_figures_are_scikit_learns() -> None:
    plots = pytest.importorskip("plots", reason="matplotlib is not installed")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.round(np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.2, 500), 0, 1), 2)  # with ties
    fpr, tpr = plots.roc_points(y, p)
    expected_fpr, expected_tpr, _ = roc_curve(y, p, drop_intermediate=False)
    assert np.allclose(fpr, expected_fpr) and np.allclose(tpr, expected_tpr)
