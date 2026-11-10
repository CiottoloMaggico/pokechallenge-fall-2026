"""The ``v11_fs`` feature set on the fixtures: the initial copy of v11's groups, the port of the
team effect and bag logit, label-aware features out of fold, the cache of their label-free
inputs, splines, state, repairs, the ``--test`` guard."""

from __future__ import annotations

import copy
import pickle
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

import usecases
from data import Dataset
from domain.battle import BattleContext
from feature_set import FeatureSet, _inner_fold_ids
from features import sets as feature_sets
from features.common import make_set
from features.core.v11_fs.forecast import battle_key
from features.core.v11_fs.label_aware import (
    SPLINE_KEYS,
    TEAM,
    TEAM_COLUMNS,
    V11BagLogit,
    V11TeamEffect,
)
from features.core.v11_fs.team_effect import TeamScores
from interfaces import Feature
from models.core.v11_stacking import StackedLogit
from models.core.v11_team import TeamEffect
from settings import Settings
from tests.features.test_v11_feature import FIXTURES, same, split

GROUPS = ["s", "f", "t", "a", "e", "m", "w", "x", "d", "v", "z", "c", "q", "spl", "stack", "tm"]
LABEL_AWARE = ("V11BagLogit", "V11TeamEffect")
BAGS = ("bs_", "bh_", "bm_")

Run = tuple[Dataset, Dataset, FeatureSet[BattleContext], pd.DataFrame]


def fitted(train: Dataset, name: str = "v11_fs") -> tuple[FeatureSet[BattleContext], pd.DataFrame]:
    features = feature_sets.build(name)
    x = features.fit_transform(
        train.contexts, train.labels.tolist(), (), train.duplicate_groups().tolist()
    )
    return features, x


@pytest.fixture(scope="module", params=["rule", "none"])
def run(request: pytest.FixtureRequest) -> Run:
    train, new = split(request.param)
    return train, new, *fitted(train)


@pytest.fixture(scope="module")
def frozen() -> tuple[FeatureSet[BattleContext], pd.DataFrame]:
    """The frozen ``v11`` set on the same battles and repair (``rule``)."""
    train, _ = split("rule")
    return fitted(train, "v11")


def feature(features: FeatureSet[BattleContext], fid: str) -> Feature[BattleContext]:
    return features._entries[fid].feature


def inner_folds(train: Dataset) -> list[int]:
    """The inner folds of ``FeatureSet.fit_transform`` (5 folds, seed 0, duplicate groups)."""
    return _inner_fold_ids(len(train), train.duplicate_groups().tolist(), 5, 0)


def values(f: Feature[BattleContext], battles: Sequence[BattleContext]) -> np.ndarray:
    f.prepare(battles)
    return np.array([list(f(c).values()) for c in battles])


def test_the_columns_and_no_raw_bags_or_keys(run: Run) -> None:
    _, new, features, x = run
    assert list(dict.fromkeys(c.split("_")[0] for c in x.columns)) == GROUPS
    assert not [c for c in x.columns if c.startswith((*BAGS, "k_"))]
    assert x.columns[-10:].tolist() == ["stack_logit_0", *TEAM_COLUMNS]
    assert len(x.filter(like="spl_").columns) == 6 * len(SPLINE_KEYS)
    assert list(features.to_frame(new.contexts).columns) == list(x.columns)
    aware = [f for f in features.feature_ids if feature(features, f).uses_labels]
    assert tuple(aware) == LABEL_AWARE
    assert all(feature(features, f).inner_oof for f in aware)


def test_the_label_free_groups_are_a_bitwise_copy_of_v11(
    frozen: tuple[FeatureSet[BattleContext], pd.DataFrame],
) -> None:
    train, new = split("rule")
    features, x = fitted(train)
    v11, x_v11 = frozen
    groups = [c for c in x_v11.columns if not c.startswith((*BAGS, "k_"))]
    assert same(x.loc[:, groups], x_v11.loc[:, groups])
    found, expected = features.to_frame(new.contexts), v11.to_frame(new.contexts)
    assert same(found.loc[:, groups], expected.loc[:, groups])


def test_the_team_scores_are_the_frozen_team_effect_on_a_synthetic_table() -> None:
    rng = np.random.default_rng(0)
    teams = [frozenset(rng.choice(30, 6, replace=False).tolist()) for _ in range(25)]
    rows = []
    for _ in range(400):
        t1, t2 = teams[rng.integers(25)], teams[rng.integers(25)]
        shown = sorted(t2)[: rng.integers(3, 7)]
        row = {"k_p1team": float(hash(tuple(sorted(t1))) % 10**9)}
        row["k_p2team"] = float(hash(tuple(sorted(t2))) % 10**9) if len(shown) == 6 else np.nan
        row["k_p2part"] = float(hash(tuple(shown)) % 10**9) if len(shown) == 5 else np.nan
        for side, team in (("1", sorted(t1)), ("2", sorted(t2))):
            for k in range(6):
                sub = [s for j, s in enumerate(team) if j != k]
                full = side == "1" or len(shown) == 6
                row[f"k_p{side}sub{k}"] = float(hash(tuple(sub)) % 10**9) if full else np.nan
        for k in range(6):
            row[f"k_p1sp{k}"] = float(sorted(t1)[k])
            row[f"k_p2sp{k}"] = float(shown[k]) if k < len(shown) else np.nan
        rows.append(row)
    x = pd.DataFrame(rows).assign(stack_logit_full=rng.normal(size=400))
    y = rng.integers(0, 2, 400)
    expected = TeamEffect(**TEAM).fit_transform(x, y)
    scores, found = TeamScores("stack_logit_full", **TEAM), {}
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(x, y):
        for name, v in scores.encode(x.iloc[va], scores.fit(x.iloc[tr], y[tr])).items():
            found.setdefault(name, np.zeros(len(x)))[va] = v
    assert np.array_equal(expected[list(TEAM_COLUMNS)].to_numpy(), pd.DataFrame(found).to_numpy())


def test_the_bag_logit_and_team_effect_are_the_frozen_steps_on_v11s_table(
    frozen: tuple[FeatureSet[BattleContext], pd.DataFrame],
) -> None:
    """Fitted on the same battles, the features score new battles as the frozen members' steps
    (bag ``StackedLogit``; full ``StackedLogit`` then ``TeamEffect``) on v11's table."""
    train, new = split("rule")
    v11, x_v11 = frozen
    y = train.labels
    x_new = v11.to_frame(new.contexts)
    bags = StackedLogit(prefixes=BAGS, C=0.01, name="stack_logit_0").fit(x_v11, y)
    full = StackedLogit(exclude=("k_",), C=0.003, name="stack_logit_full", keep=True)
    effect = TeamEffect(**TEAM).fit(full.fit_transform(x_v11, y), y)
    expected = effect.transform(full.transform(x_new))[list(TEAM_COLUMNS)].to_numpy()
    bag_logit, team = V11BagLogit(), V11TeamEffect()
    bag_logit.fit(train.contexts, y.tolist())
    team.fit(train.contexts, y.tolist())
    found = values(bag_logit, new.contexts)[:, 0]
    assert np.allclose(found, bags.transform(x_new)["stack_logit_0"], rtol=0, atol=1e-12)
    assert np.allclose(values(team, new.contexts), expected, rtol=0, atol=1e-12)


def test_a_training_row_comes_from_a_copy_fitted_without_its_inner_fold(run: Run) -> None:
    train, _, features, x = run
    folds = inner_folds(train)
    inside = [i for i, f in enumerate(folds) if f == 0]
    outside = [i for i, f in enumerate(folds) if f != 0]
    labels = train.labels.tolist()
    for fid in LABEL_AWARE:
        twin = copy.deepcopy(feature(features, fid))
        twin.reset()
        twin.fit([train.contexts[i] for i in outside], [labels[i] for i in outside])
        names = list(twin.output_names or ())
        found = np.array([list(twin(train.contexts[i]).values()) for i in inside])
        assert np.array_equal(x.loc[inside, names].to_numpy(), found)
        full = values(feature(features, fid), [train.contexts[i] for i in inside])
        assert not np.array_equal(found, full)  # not the in-sample values


def label_aware_rows(
    features: FeatureSet[BattleContext], train: Dataset, labels: list[int]
) -> pd.DataFrame:
    """The label-aware columns of ``fit_transform`` with copies of the fitted features (their
    label-free inputs, fitted on these battles, are reused)."""
    twins: list[Feature[BattleContext]] = []
    for fid in LABEL_AWARE:
        twin = copy.deepcopy(feature(features, fid))
        twin.reset()
        twins.append(twin)
    subset = make_set("label_aware", None, twins)
    return subset.fit_transform(train.contexts, labels, (), train.duplicate_groups().tolist())


def test_a_row_ignores_its_own_label_but_not_the_others(run: Run) -> None:
    train, _, features, x = run
    labels = train.labels.tolist()
    folds = inner_folds(train)
    inside = [i for i, f in enumerate(folds) if f == 0]
    own = [1 - y if folds[i] == 0 else y for i, y in enumerate(labels)]
    flipped_own = label_aware_rows(features, train, own)
    assert np.array_equal(
        flipped_own.loc[inside].to_numpy(), x.loc[inside, flipped_own.columns].to_numpy()
    )
    others = [1 - y if folds[i] != 0 else y for i, y in enumerate(labels)]
    flipped_others = label_aware_rows(features, train, others)
    for fid in LABEL_AWARE:
        names = list(feature(features, fid).output_names or ())
        assert not np.array_equal(
            flipped_others.loc[inside, names].to_numpy(), x.loc[inside, names].to_numpy()
        )


@pytest.mark.parametrize("kind", [V11BagLogit, V11TeamEffect])
def test_a_clone_refitted_on_other_battles_does_not_reuse_the_inputs(
    run: Run, kind: type[V11BagLogit] | type[V11TeamEffect]
) -> None:
    train, _, _, _ = run
    battles, labels = list(train.contexts), train.labels.tolist()
    first = kind()
    first.fit(battles[:100], labels[:100])
    reused = make_set("one", None, [first]).clone()._entries[first.feature_id].feature
    assert isinstance(reused, kind)
    reused.fit(battles[:60], labels[:60])
    assert reused.inputs.keys == first.inputs.keys  # a subset: reused
    other = make_set("one", None, [first]).clone()._entries[first.feature_id].feature
    assert isinstance(other, kind)
    other.fit(battles[50:], labels[50:])
    wanted = {battle_key(c) for c in battles[50:]}
    assert other.inputs.keys == wanted
    fresh = kind()
    fresh.fit(battles[50:], labels[50:])
    assert np.array_equal(values(other, battles[:50]), values(fresh, battles[:50]))
    if kind is V11TeamEffect:
        assert set(other.inputs.features[-1]._train_rows) == wanted  # type: ignore[attr-defined]


def test_the_splines_expand_the_keys_of_the_table(run: Run) -> None:
    _, _, _, x = run
    pipe = Pipeline(
        [("imp", SimpleImputer(strategy="median")), ("s", SplineTransformer(n_knots=4, degree=3))]
    )
    found = x.filter(like="spl_").to_numpy()
    assert np.array_equal(pipe.fit_transform(x[list(SPLINE_KEYS)]), found)


def test_a_saved_state_scores_new_battles_like_the_fitted_set(run: Run) -> None:
    _, new, features, _ = run
    restored = feature_sets.build("v11_fs")
    restored.set_state(pickle.loads(pickle.dumps(features.get_state())))
    assert same(restored.to_frame(new.contexts), features.to_frame(new.contexts))


def test_the_set_refuses_a_turn_and_the_commands_refuse_test_battles() -> None:
    with pytest.raises(ValueError, match="fixed at the last turn"):
        feature_sets.build("v11_fs@t10")
    settings = Settings(FIXTURES, test=FIXTURES, repair="none")
    for command in (usecases.train, usecases.evaluate):
        with pytest.raises(SystemExit, match="drop --test"):
            command(settings, ["v11_fs"], ["logreg"])
