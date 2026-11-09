"""The ``v11`` feature set on the fixtures: a saved state scores new battles like the fitted one."""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import pytest

from data import Dataset, load_records
from domain.battle import BattleContext
from domain.pokedex import PokedexBuilder
from domain.records import without_label
from features import sets as feature_sets
from features.core.v11.feature import V11Features

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "battles.jsonl"
N_TRAIN = 150

Contexts = tuple[BattleContext, ...]


@pytest.fixture(scope="module")
def split() -> tuple[Dataset, Dataset]:
    records = load_records(FIXTURES)
    pokedex = PokedexBuilder().add_records(records[:N_TRAIN]).build()
    new = [without_label(r) for r in records[N_TRAIN:]]
    return Dataset.from_records(records[:N_TRAIN], pokedex), Dataset.from_records(new, pokedex)


@pytest.fixture(scope="module")
def fitted(split: tuple[Dataset, Dataset]) -> V11Features:
    train, _ = split
    feature = V11Features()
    feature.fit(train.contexts, train.labels.tolist())
    return feature


def same(a: dict[str, float], b: dict[str, float]) -> bool:
    return a.keys() == b.keys() and all(
        a[k] == b[k] or (math.isnan(a[k]) and math.isnan(b[k])) for k in a
    )


def test_a_saved_state_scores_a_new_battle_like_the_fitted_feature(
    split: tuple[Dataset, Dataset], fitted: V11Features
) -> None:
    train, new = split
    restored = V11Features()
    restored.set_state(pickle.loads(pickle.dumps(fitted.get_state())))
    fitted.prepare(new.contexts)
    for context in new.contexts:
        # the restored feature scores each battle alone, the fitted one in one batch
        assert same(dict(restored.compute(context)), dict(fitted.compute(context)))
    for context in train.contexts[:10]:
        assert same(dict(restored.compute(context)), dict(fitted.compute(context)))


def test_the_columns_follow_the_matrix_order(fitted: V11Features) -> None:
    columns = fitted.output_names
    assert columns is not None and "cut" not in columns
    prefixes = [c.split("_")[0] for c in columns]
    first = list(dict.fromkeys(prefixes))
    assert first[:2] == ["s", "f"] and first[-2:] == ["c", "q"]


def test_the_set_refuses_a_turn() -> None:
    with pytest.raises(ValueError, match="fixed at the last turn"):
        feature_sets.build("v11@t10")
