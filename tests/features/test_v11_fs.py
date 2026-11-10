"""The ``v11_fs`` feature set on the fixtures: the initial copy of v11's groups, the port of the
team effect and bag logit, label-aware table features out of fold, splines, state, repairs, the
``--test`` guard."""

from __future__ import annotations

import copy
import pickle
from collections.abc import Mapping, Sequence
from typing import Any

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
from staged_feature_set import StagedFeatureSet, TableFeature
from tests.features.test_v11_feature import FIXTURES, same, split

GROUPS = ["s", "f", "t", "a", "e", "m", "w", "x", "d", "v", "z", "c", "q", "spl", "stack", "tm"]
LABEL_AWARE = ("V11BagLogit", "V11TeamEffect")
LABEL_COLUMNS = ["stack_logit_0", "stack_logit_full", *TEAM_COLUMNS]
BAGS = ("bs_", "bh_", "bm_")

Run = tuple[Dataset, Dataset, Any, pd.DataFrame]


def fitted(train: Dataset, name: str = "v11_fs") -> tuple[Any, pd.DataFrame]:
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


def table_feature(features: StagedFeatureSet[BattleContext], fid: str) -> TableFeature:
    (found,) = [f for f in features.stage if f.feature_id == fid]
    return found


def inner_folds(train: Dataset) -> list[int]:
    """The inner folds of the set (5 folds, seed 0, duplicate groups)."""
    return _inner_fold_ids(len(train), train.duplicate_groups().tolist(), 5, 0)


def test_the_columns_and_no_raw_bags_or_keys(run: Run) -> None:
    _, new, features, x = run
    assert isinstance(features, StagedFeatureSet)
    assert list(dict.fromkeys(c.split("_")[0] for c in x.columns)) == GROUPS
    assert not [c for c in x.columns if c.startswith((*BAGS, "k_"))]
    assert x.columns[-11:].tolist() == LABEL_COLUMNS
    assert len(x.filter(like="spl_").columns) == 6 * len(SPLINE_KEYS)
    assert list(features.to_frame(new.contexts).columns) == list(x.columns)
    assert features.declared_columns is None  # the groups do not declare their columns
    aware = [f.feature_id for f in features.stage if f.uses_labels]
    assert tuple(aware) == LABEL_AWARE
    assert all(table_feature(features, f).inner_oof for f in aware)
    assert features.base.feature_ids[7:9] == ["V11Bags", "V11TeamKeys"]


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
    """Fitted on v11's table, the features score new battles as the frozen members' steps
    (bag ``StackedLogit``; full ``StackedLogit``, whose logit is output, then ``TeamEffect``)."""
    train, new = split("rule")
    v11, x_v11 = frozen
    y = train.labels
    x_new = v11.to_frame(new.contexts)
    bags = StackedLogit(prefixes=BAGS, C=0.01, name="stack_logit_0").fit(x_v11, y)
    full = StackedLogit(exclude=("k_",), C=0.003, name="stack_logit_full", keep=True)
    effect = TeamEffect(**TEAM).fit(full.fit_transform(x_v11, y), y)
    expected = effect.transform(full.transform(x_new))[LABEL_COLUMNS[1:]].to_numpy()
    bag_logit = V11BagLogit().fit(x_v11, y.tolist())
    team = V11TeamEffect().fit(x_v11, y.tolist())
    found = bag_logit.transform(x_new)[:, 0]
    assert np.array_equal(found, bags.transform(x_new)["stack_logit_0"].to_numpy())
    assert np.array_equal(team.transform(x_new), expected)


def test_a_training_row_comes_from_a_copy_fitted_without_its_inner_fold(run: Run) -> None:
    train, _, features, x = run
    folds = inner_folds(train)
    inside = [i for i, f in enumerate(folds) if f == 0]
    outside = [i for i, f in enumerate(folds) if f != 0]
    labels = train.labels.tolist()
    table = features.base.to_frame(train.contexts)
    for fid in LABEL_AWARE:
        feature = table_feature(features, fid)
        twin = copy.deepcopy(feature)
        twin.reset()
        twin.fit(table.iloc[outside].reset_index(drop=True), [labels[i] for i in outside])
        found = twin.transform(table.iloc[inside].reset_index(drop=True))
        assert np.array_equal(x.loc[inside, list(twin.output_names)].to_numpy(), found)
        full = feature.transform(table.iloc[inside])
        assert not np.array_equal(found, full)  # not the in-sample values


class Lookup(Feature[BattleContext]):
    """A precomputed first-stage row by battle key (a cheap first stage for the label tests)."""

    def __init__(self, table: pd.DataFrame, battles: Sequence[BattleContext]) -> None:
        super().__init__()
        self.columns = tuple(table.columns)
        self.rows = {battle_key(c): r for c, r in zip(battles, table.to_numpy(), strict=True)}

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.columns

    def compute(self, battle: BattleContext) -> Mapping[str, float]:
        return dict(zip(self.columns, self.rows[battle_key(battle)].tolist(), strict=True))


def label_aware_rows(features: Any, train: Dataset, labels: list[int]) -> pd.DataFrame:
    """The label-aware columns of the set's ``fit_transform`` on the same first-stage table."""
    lookup = make_set(
        "lookup", None, [Lookup(features.base.to_frame(train.contexts), train.contexts)]
    )
    staged = StagedFeatureSet("label_aware", lookup, [V11BagLogit(), V11TeamEffect()])
    x = staged.fit_transform(train.contexts, labels, (), train.duplicate_groups().tolist())
    return x.loc[:, LABEL_COLUMNS]


def test_a_row_ignores_its_own_label_but_not_the_others(run: Run) -> None:
    train, _, features, x = run
    labels = train.labels.tolist()
    folds = inner_folds(train)
    inside = [i for i, f in enumerate(folds) if f == 0]
    assert same(label_aware_rows(features, train, labels), x.loc[:, LABEL_COLUMNS])
    own = [1 - y if folds[i] == 0 else y for i, y in enumerate(labels)]
    flipped_own = label_aware_rows(features, train, own)
    assert np.array_equal(
        flipped_own.loc[inside].to_numpy(), x.loc[inside, flipped_own.columns].to_numpy()
    )
    others = [1 - y if folds[i] != 0 else y for i, y in enumerate(labels)]
    flipped_others = label_aware_rows(features, train, others)
    for names in (["stack_logit_0"], ["stack_logit_full"], list(TEAM_COLUMNS)):
        assert not np.array_equal(
            flipped_others.loc[inside, names].to_numpy(), x.loc[inside, names].to_numpy()
        )


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
