"""The ``v11`` feature set on the fixtures: matrix order, out-of-fold forecast, repairs, state."""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from addons import load_repair
from data import Dataset, load_records
from domain.pokedex import PokedexBuilder
from domain.records import without_label
from feature_set import FeatureSet
from features import sets as feature_sets
from features.core.v11.forecast import OUTPUTS, V11Forecast, battle_key

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "battles.jsonl"
N_TRAIN = 150
GROUPS = ["s", "f", "t", "a", "e", "m", "w", "x", "bs", "bh", "bm", "k", "d", "v", "z", "c", "q"]


def split(repair: str) -> tuple[Dataset, Dataset]:
    records = load_records(FIXTURES)
    pokedex = PokedexBuilder().add_records(records[:N_TRAIN]).build()
    chosen = load_repair(repair, pokedex, feature_sets.type_chart())
    new = [without_label(r) for r in records[N_TRAIN:]]
    return (
        Dataset.from_records(records[:N_TRAIN], pokedex, repair=chosen),
        Dataset.from_records(new, pokedex, repair=chosen),
    )


def fitted(train: Dataset) -> tuple[FeatureSet, pd.DataFrame]:
    features = feature_sets.build("v11")
    x = features.fit_transform(
        train.contexts, train.labels.tolist(), (), train.duplicate_groups().tolist()
    )
    return features, x


@pytest.fixture(scope="module", params=["rule", "none"])
def run(request: pytest.FixtureRequest) -> tuple[Dataset, Dataset, FeatureSet, pd.DataFrame]:
    train, new = split(request.param)
    features, x = fitted(train)
    return train, new, features, x


def same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return list(a.columns) == list(b.columns) and np.array_equal(
        a.to_numpy(), b.to_numpy(), equal_nan=True
    )


def test_the_columns_follow_the_matrix_order(
    run: tuple[Dataset, Dataset, FeatureSet, pd.DataFrame],
) -> None:
    _, new, features, x = run
    groups = list(dict.fromkeys(c.split("_")[0] for c in x.columns))
    assert groups == GROUPS and "cut" not in x.columns
    assert tuple(x.columns[-6:]) == OUTPUTS
    assert list(features.to_frame(new.contexts).columns) == list(x.columns)
    assert features.feature_ids[0] == "V11Static" and features.feature_ids[-1] == "V11Forecast"


def test_every_v11_feature_is_label_free(
    run: tuple[Dataset, Dataset, FeatureSet, pd.DataFrame],
) -> None:
    _, _, features, _ = run
    assert not features.uses_labels


def test_the_forecast_is_out_of_fold_on_its_training_battles(
    run: tuple[Dataset, Dataset, FeatureSet, pd.DataFrame],
) -> None:
    train, _, features, x = run
    forecast = features._entries["V11Forecast"].feature
    assert isinstance(forecast, V11Forecast)
    for k, context in enumerate(train.contexts[:20]):
        assert x.loc[k, list(OUTPUTS)].tolist() == forecast._train_rows[battle_key(context)]
    # a training battle seen as new gets the regressors' prediction, not its out-of-fold row
    twin = forecast._train_rows.pop(battle_key(train.contexts[0]))
    predicted = list(forecast.compute(train.contexts[0]).values())
    forecast._train_rows[battle_key(train.contexts[0])] = twin
    assert predicted != twin


def test_a_saved_state_scores_new_battles_like_the_fitted_set(
    run: tuple[Dataset, Dataset, FeatureSet, pd.DataFrame],
) -> None:
    train, new, features, _ = run
    restored = feature_sets.build("v11")
    restored.set_state(pickle.loads(pickle.dumps(features.get_state())))
    assert same(restored.to_frame(new.contexts), features.to_frame(new.contexts))
    first = restored.to_frame(train.contexts[:10])
    assert same(first, features.to_frame(train.contexts[:10]))


def test_the_values_are_deterministic() -> None:
    train, new = split("rule")
    a, b = fitted(train), fitted(train)
    assert same(a[1], b[1])
    assert same(a[0].to_frame(new.contexts), b[0].to_frame(new.contexts))
    assert all(not math.isinf(v) for v in a[1].to_numpy().ravel())


def test_the_set_refuses_a_turn() -> None:
    with pytest.raises(ValueError, match="fixed at the last turn"):
        feature_sets.build("v11@t10")
