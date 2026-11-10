"""v11 models: the team effect on hand-made tables, the stacked logit is out of fold, generic
members on tables without v11's columns, the logit blend."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

import models
from interfaces import Model
from models.core.estimator import EstimatorModel
from models.core.v11_blend import LogitBlend
from models.core.v11_stacking import SplineLogit, StackedLogit
from models.core.v11_team import TeamEffect

NAN = np.nan
A, B, C = (1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 7), (10, 11, 12, 13, 14, 15)


def team_table(k1: list[float], k2: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"k_p1team": k1, "k_p2team": k2, "stack_logit_full": 0.0})


def test_team_effect_counts_p1_and_p2_teams() -> None:
    train = team_table([1, 1, 2], [2, NAN, 1])
    y = np.array([1, 1, 0])
    effect = TeamEffect(a=1.0).fit(train, y)
    out = effect.transform(team_table([1, 1], [2, NAN]))
    # residuals y - 1/2: team 1 gets +.5 +.5 (P1) and +.5 (P2 of a lost battle), team 2 gets -1
    assert out["tm"].tolist() == pytest.approx([1.5 / 4 + 1.0 / 3, 1.5 / 4])
    assert out["tm_n1"].tolist() == [3, 3]
    assert out["tm_n2"].tolist() == [2, 0]
    assert list(out.columns) == ["stack_logit_full", "tm", "tm_n1", "tm_n2"]


def test_link_matches_a_five_species_team_to_the_only_candidate() -> None:
    def table(k1: list[float], subs: list[list[float]], part: list[float]) -> pd.DataFrame:
        x = team_table(k1, [NAN] * len(k1))
        for side in ("1", "2"):
            for k in range(6):
                x[f"k_p{side}sub{k}"] = [row[k] if side == "1" else NAN for row in subs]
        x["k_p2part"] = part
        return x

    train = table([1, 2], [[11, 12, 13, 14, 15, 16], [21, 22, 23, 24, 25, 26]], [NAN, NAN])
    effect = TeamEffect(a=1.0, link=True, pair=True).fit(train, np.array([1, 0]))
    out = effect.transform(table([2], [[21, 22, 23, 24, 25, 26]], [13]))
    assert out["tm_n2"].tolist() == [1]
    assert out["tm"].tolist() == pytest.approx([-0.5 / 2 - 0.5 / 2])
    assert out["tm_npair"].tolist() == [0]


def species_table(p1: list[tuple[int, ...]], p2: list[tuple[int, ...]], k1, k2) -> pd.DataFrame:
    x = team_table(k1, k2)
    for k in range(6):
        x[f"k_p1sp{k}"] = [float(t[k]) for t in p1]
        x[f"k_p2sp{k}"] = [float(t[k]) if k < len(t) else NAN for t in p2]
    return x


def test_soft_matching_and_raw_ratings_of_a_five_species_team() -> None:
    train = species_table([A, A, B], [C, (20, 21), (20, 21)], [1, 1, 2], [3, NAN, NAN])
    effect = TeamEffect(a=1.0, soft=True, raw=True).fit(train, np.array([1, 1, 0]))
    out = effect.transform(species_table([C], [(1, 2, 3, 4, 5)], [3], [NAN]))
    # candidates A (played twice, residual sum 1) and B (once, -0.5), weighted 2 : 1
    soft = -(2 * (1.0 / 3) + 1 * (-0.5 / 2)) / 3
    assert out["tm_soft"].tolist() == pytest.approx([soft])
    assert out["tm_nsoft"].tolist() == [3]
    assert out["tm"].tolist() == pytest.approx([-0.5 / 2 + soft])
    assert out["tm_raw1"].tolist() == pytest.approx([-0.5 / 3])
    assert out["tm_raw2"].tolist() == pytest.approx([(2 * (1.0 / 4) + 1 * (-0.5 / 3)) / 3])


def random_table(n: int = 120) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f_a", "bs_b", "bh_c"])
    y = (x["f_a"] + rng.normal(size=n) > 0).astype(int).to_numpy()
    return x, y


def test_stacked_logit_fit_transform_is_out_of_fold() -> None:
    x, y = random_table()
    stack = StackedLogit(exclude=("bh_",), C=0.05, name="s")
    out = stack.fit_transform(x, y)
    assert list(out.columns) == ["bh_c", "s"] and stack.columns_ == ["f_a", "bs_b"]
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(x, y):
        fold = StackedLogit(prefixes=("f_", "bs_"), C=0.05).fit(x.iloc[tr], y[tr])
        expected = fold.model_.decision_function(x.iloc[va][["f_a", "bs_b"]])
        assert np.array_equal(out["s"].to_numpy()[va], expected)
    full = stack.model_.decision_function(x[["f_a", "bs_b"]])
    assert np.array_equal(stack.transform(x)["s"].to_numpy(), full)
    assert not np.array_equal(out["s"].to_numpy(), full)


def test_steps_without_their_columns_are_the_identity() -> None:
    x, y = random_table()
    assert StackedLogit(prefixes=("k_",)).fit_transform(x, y) is x
    effect = TeamEffect(drop_base=True, base="f_a")
    assert effect.fit_transform(x, y) is x and effect.transform(x) is x
    spline = SplineLogit(C=0.1, keys=["f_a", "f_missing"]).fit(x, y)
    assert [name for name, _ in spline.pipeline_.steps][0] == "ct"
    plain = SplineLogit(C=0.1, keys=["f_missing"]).fit(x, y)
    assert [name for name, _ in plain.pipeline_.steps][0] == "imp"


@pytest.mark.parametrize("name", ["v11_lr_spline", "v11_lgbm"])
def test_members_fit_a_table_without_v11_columns(name: str, tmp_path) -> None:
    x, y = random_table()
    member = models.build(name, **({"n_estimators": 20} if name == "v11_lgbm" else {}))
    fitted = member.clone().fit(x, y)
    p = fitted.predict_proba(x)
    assert p.shape == (len(x),) and ((0 <= p) & (p <= 1)).all()
    fitted.save(tmp_path / "m.pkl")
    assert np.array_equal(Model.load(tmp_path / "m.pkl").predict_proba(x), p)
    assert not member.clone().is_fitted


def test_the_member_pipelines_follow_the_original() -> None:
    booster, spline_member = models.build("v11_xgb"), models.build("v11_lr_spline")
    assert isinstance(booster, EstimatorModel) and isinstance(spline_member, EstimatorModel)
    steps = [name for name, _ in booster.estimator.steps]
    assert steps == ["stack_full", "stack0", "team", "model"]
    spline = spline_member.estimator
    assert [name for name, _ in spline.steps] == ["team_base", "team", "model"]
    assert spline.steps[1][1].drop_base and spline.steps[0][1].spline_keys[0] == "f_d_fainted"


def test_blend_is_the_weighted_logit_average_of_its_members() -> None:
    x, y = random_table()
    members = [models.build("v11_lr_spline", C=0.1), models.build("v11_lr_spline", C=0.1)]
    blend = LogitBlend("b", members, [0.3, 0.7]).fit(x, y)
    p = np.clip(members[0].predict_proba(x), 1e-6, 1 - 1e-6)
    assert np.allclose(blend.predict_proba(x), p)
    doubled = LogitBlend("b", members, [1.0, 1.0]).fit(x, y).predict_proba(x)
    assert np.allclose(doubled, 1 / (1 + ((1 - p) / p) ** 2))
    assert not blend.clone().is_fitted
