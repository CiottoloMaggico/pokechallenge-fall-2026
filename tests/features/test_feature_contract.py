"""Contract of every feature (REFACTORING.md §4.1), on the fixtures.

A feature gives exactly its declared columns as floats, the same values every time and under any
``PYTHONHASHSEED``, survives a save / load of its state and never sees the label.

Run as a script, this file writes the output of every feature to a JSON file: the hash seed is
fixed when a process starts, so comparing two seeds takes two processes.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from functools import cache
from pathlib import Path

import pytest

import feature_catalog
from data import Dataset
from domain.battle import BattleContext
from domain.pokedex import Pokedex, PokedexBuilder
from domain.records import read_records, without_label
from features.bag_features import BagLogit
from features.encoding_features import TeamEffect
from features.knowledge_features import TeamKnowledgeFeatures
from features.matchup_features import TeamMatchup, ttk_advantage
from features.material_features import hp_dealt
from features.montecarlo_features import MonteCarloSimulation
from features.team_features import bst_hp_left
from features.trajectory_features import hp_balance_delta, hp_balance_mean, ko_balance_mean
from features.turn_features import AtTurn
from features.wincon_features import (
    healers,
    phazers,
    setup_users,
    sleepers,
    spikes_advantage,
)
from interfaces import Feature
from knockout_detector import DetectorRepair, KnockoutDetector, default_classifier

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "battles.jsonl"
CHART = feature_catalog.type_chart()
EXCLUDED_SETS = {"compact_v5_v11"}  # needs the external v11 repository
HASH_SEEDS = ("0", "1")
DETECTOR_TREES = 30  # enough to score every switch; the real detector takes minutes
LABEL_CHECK_BATTLES = 20

Row = list[tuple[str, str]]


def unregistered_features() -> list[Feature[BattleContext]]:
    """Features only the experiments use: in no feature set, but under the same contract."""
    return [
        TeamKnowledgeFeatures(CHART),
        TeamEffect(),
        BagLogit(),
        TeamMatchup(CHART),
        MonteCarloSimulation(CHART, runs=2),
        AtTurn("wc_healers", healers),
        AtTurn("wc_setup", setup_users),
        AtTurn("wc_phazers", phazers),
        AtTurn("wc_sleepers", sleepers),
        AtTurn("wc_spikes_adv", spikes_advantage),
        AtTurn("bst_hp_left", bst_hp_left),
        AtTurn("act_ttk_adv", ttk_advantage(CHART)),
        AtTurn("hp_balance_mean", hp_balance_mean),
        AtTurn("ko_balance_mean", ko_balance_mean),
        AtTurn("hp_balance_delta", hp_balance_delta),
        AtTurn("hp_dealt", hp_dealt),
    ]


def new_features() -> dict[str, Feature[BattleContext]]:
    """A new unfitted copy of every feature; one id can have several output selections."""
    found = list(unregistered_features())
    for name in sorted(set(feature_catalog.FEATURE_SETS) - EXCLUDED_SETS):
        features = feature_catalog.build(name)
        found.extend(features._entries[fid].feature for fid in features.feature_ids)
    return {f"{f.feature_id}:{'+'.join(f.output_names or ())}": f for f in found}


KEYS = sorted(new_features())


@cache
def pokedex() -> Pokedex:
    return PokedexBuilder().add_records(read_records(FIXTURES)).build()


@cache
def repair() -> DetectorRepair:
    classifier = default_classifier().set_params(max_iter=DETECTOR_TREES)
    detector = KnockoutDetector(pokedex().find, CHART, classifier=lambda: classifier)
    return DetectorRepair(detector.fit(without_label(r) for r in read_records(FIXTURES)), pokedex())


@cache
def battles() -> Dataset:
    return Dataset.from_jsonl(FIXTURES, pokedex(), repair=repair())


@cache
def fitted(key: str) -> Feature[BattleContext]:
    feature = new_features()[key]
    if feature.requires_fit:
        labels = battles().labels.tolist() if feature.uses_labels else None
        feature.fit(battles().contexts, labels)
    return feature


def outputs(
    feature: Feature[BattleContext], contexts: tuple[BattleContext, ...] | None = None
) -> list[Row]:
    """Every row in order, with exact values: ``float.hex`` also makes NaN equal to itself."""
    rows = battles().contexts if contexts is None else contexts
    return [[(k, float(v).hex()) for k, v in feature.compute(c).items()] for c in rows]


@pytest.fixture(scope="module")
def outputs_by_hash_seed(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, list[Row]]]:
    folder = tmp_path_factory.mktemp("hash_seeds")
    paths = [folder / f"{seed}.json" for seed in HASH_SEEDS]
    workers = [
        subprocess.Popen(
            [sys.executable, __file__, str(path)],
            env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT)},
            cwd=ROOT,
        )
        for seed, path in zip(HASH_SEEDS, paths, strict=True)
    ]
    assert [worker.wait() for worker in workers] == [0, 0]
    return [json.loads(path.read_text()) for path in paths]


@pytest.mark.parametrize("key", KEYS)
def test_identical_under_two_hash_seeds(
    key: str, outputs_by_hash_seed: list[dict[str, list[Row]]]
) -> None:
    first, second = (result[key] for result in outputs_by_hash_seed)
    assert first == second


@pytest.mark.parametrize("key", KEYS)
def test_gives_exactly_the_declared_columns_as_floats(key: str) -> None:
    feature = fitted(key)
    for context in battles().contexts:
        row = feature.compute(context)
        assert row and all(isinstance(v, float) for v in row.values())
        if feature.output_names is not None:
            assert tuple(row) == feature.output_names


@pytest.mark.parametrize("key", KEYS)
def test_same_battle_same_values(key: str) -> None:
    feature = fitted(key)
    assert outputs(feature) == outputs(feature)


@pytest.mark.parametrize("key", KEYS)
def test_saved_state_gives_the_same_values(key: str) -> None:
    feature = fitted(key)
    if not feature.requires_fit:
        pytest.skip("no state")
    restored = new_features()[key]
    restored.set_state(pickle.loads(pickle.dumps(feature.get_state())))
    assert restored.is_fitted
    assert outputs(restored) == outputs(feature)


@pytest.mark.parametrize("key", KEYS)
def test_values_do_not_depend_on_the_label(key: str) -> None:
    records = [json.loads(line) for line in FIXTURES.open()][:LABEL_CHECK_BATTLES]
    flipped = [{**r, "player_won": 1 - r["player_won"]} for r in records]
    feature = fitted(key)
    assert outputs(feature, Dataset.from_records(records, pokedex(), repair=repair()).contexts) == (
        outputs(feature, Dataset.from_records(flipped, pokedex(), repair=repair()).contexts)
    )


def test_contexts_never_carry_the_label() -> None:
    assert all("player_won" not in c.record for c in battles().contexts)


def test_every_family_of_features_is_checked() -> None:
    ids = {key.partition(":")[0] for key in KEYS}
    assert {"team_knowledge", "pool_bulk_spd", "team_rating", "team_encoding"} <= ids


if __name__ == "__main__":
    all_outputs = {key: outputs(fitted(key)) for key in KEYS}
    Path(sys.argv[1]).write_text(json.dumps(all_outputs))
