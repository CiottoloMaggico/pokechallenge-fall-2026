"""Contract of every feature (REFACTORING.md §4.1), on the fixtures.

A feature gives exactly its declared columns as floats, the same values every time and under any
``PYTHONHASHSEED``, survives a save / load of its state and never sees the label. The features of
the detector's sets (``features.transitions``) run on the switches of the fixtures and must not
read the recorded faint, their label.

Run as a script, this file writes the output of every feature to a JSON file: the hash seed is
fixed when a process starts, so comparing two seeds takes two processes.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
import subprocess
import sys
from functools import cache
from pathlib import Path

import pytest

from addons.knockouts_model import ModelKnockoutRepair
from data import Dataset
from detector import DetectorTask, load_task
from domain.battle import BattleContext
from domain.damage import DamageCalculator
from domain.pokedex import Pokedex, PokedexBuilder
from domain.records import read_records
from domain.transitions import Transition
from features import sets as feature_sets
from features import transitions
from features.core.bag import BagLogit
from features.core.encoding import TeamEffect
from features.core.knowledge import TeamKnowledgeFeatures
from features.core.matchup import TeamMatchup, ttk_advantage
from features.core.material import hp_dealt
from features.core.montecarlo import MonteCarloSimulation
from features.core.team import bst_hp_left
from features.core.trajectory import hp_balance_delta, hp_balance_mean, ko_balance_mean
from features.core.turn import AtTurn
from features.core.wincon import (
    healers,
    phazers,
    setup_users,
    sleepers,
    spikes_advantage,
)
from gym import Gym
from interfaces import Feature
from models import build

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "battles.jsonl"
CHART = feature_sets.type_chart()
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
    for name in sorted(feature_sets.FEATURE_SETS):
        features = feature_sets.build(name)
        found.extend(features._entries[fid].feature for fid in features.feature_ids)
    return {f"{f.feature_id}:{'+'.join(f.output_names or ())}": f for f in found}


KEYS = sorted(new_features())
TRANSITION_KEYS = sorted(transitions.TRANSITION_SETS)


@cache
def pokedex() -> Pokedex:
    return PokedexBuilder().add_records(read_records(FIXTURES)).build()


@cache
def task() -> DetectorTask:
    return load_task(FIXTURES, pokedex())


def switches() -> tuple[list[Transition], list[Transition]]:
    """P1's rows and P2's candidates of the fixtures."""
    return list(task().p1.items), task().p2


@cache
def repair() -> ModelKnockoutRepair:
    gym = Gym([build("hgb_ko", max_iter=DETECTOR_TREES)], [transitions.build("ko_domain30")])
    (run,) = gym.train(task().p1)
    info = {"version": "test", "threshold": 0.5, "feature_set": "ko_domain30", "model": "hgb_ko"}
    info |= {"params": {}, "train_sha256": None, "pokedex_sha256": None}
    return ModelKnockoutRepair(run, info, pokedex(), DamageCalculator(CHART))


def transition_feature(name: str) -> Feature[Transition]:
    features = transitions.build(name)
    (feature_id,) = features.feature_ids
    return features._entries[feature_id].feature


def transition_outputs(name: str, rows: list[Transition] | None = None) -> list[Row]:
    p1, p2 = switches()
    feature = transition_feature(name)
    return [
        [(k, float(v).hex()) for k, v in feature.compute(t).items()]
        for t in (p1 + p2 if rows is None else rows)
    ]


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


@pytest.mark.parametrize("name", TRANSITION_KEYS)
def test_transition_features_under_two_hash_seeds(
    name: str, outputs_by_hash_seed: list[dict[str, list[Row]]]
) -> None:
    first, second = (result[f"transitions:{name}"] for result in outputs_by_hash_seed)
    assert first == second


@pytest.mark.parametrize("name", TRANSITION_KEYS)
def test_transition_features_give_their_columns_and_ignore_the_faint(name: str) -> None:
    feature = transition_feature(name)
    p1, p2 = switches()
    assert p1 and p2
    for t in p1 + p2:
        row = feature.compute(t)
        assert tuple(row) == feature.output_names
        assert all(isinstance(v, float) for v in row.values())
    flipped = [dataclasses.replace(t, fainted=not t.fainted) for t in p1]
    assert transition_outputs(name, flipped) == transition_outputs(name, p1)


def test_contexts_never_carry_the_label() -> None:
    assert all("player_won" not in c.record for c in battles().contexts)


def test_every_family_of_features_is_checked() -> None:
    ids = {key.partition(":")[0] for key in KEYS}
    assert {"team_knowledge", "pool_bulk_spd", "team_rating", "team_encoding"} <= ids


if __name__ == "__main__":
    all_outputs = {key: outputs(fitted(key)) for key in KEYS}
    all_outputs |= {f"transitions:{name}": transition_outputs(name) for name in TRANSITION_KEYS}
    Path(sys.argv[1]).write_text(json.dumps(all_outputs))
