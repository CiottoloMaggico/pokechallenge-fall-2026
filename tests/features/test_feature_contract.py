"""Contract of every feature (``interfaces.Feature``), on the fixtures.

A feature gives exactly its declared columns as floats, the same values every time and under any
``PYTHONHASHSEED``, survives a save / load of its state and never sees the label. The features of
the detector's sets (``features.transitions``) run on the switches of the fixtures and must not
read the recorded faint, their label.

Run as a script, this file writes the output of every feature to a JSON file: the hash seed is
fixed when a process starts, so comparing two seeds takes two processes.
"""

from __future__ import annotations

import dataclasses
import faulthandler
import json
import os
import pickle
import subprocess
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from addons.knockouts_model import ModelKnockoutRepair
from data import Dataset
from detector import DetectorTask, load_task
from domain.battle import BattleContext
from domain.damage import DamageCalculator
from domain.pokedex import Missing, Pokedex, PokedexBuilder
from domain.records import read_records
from domain.switches import SwitchExtractor
from domain.transitions import Transition
from features import sets as feature_sets
from features import transitions
from features.core.encoding import TeamEffect
from features.core.matchup import ttk_advantage
from features.core.turn import AtTurn
from features.transitions import ko_v11
from gym import Gym
from interfaces import Feature
from models import build
from staged_feature_set import StagedFeatureSet

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "battles.jsonl"
CHART = feature_sets.type_chart()
HASH_SEEDS = ("0", "1")
WORKER_SECONDS = 420  # one run of every feature; a slower one dumps its stacks and fails
DETECTOR_TREES = 30  # enough to score every switch; the real detector takes minutes
LABEL_CHECK_BATTLES = 20

Row = list[tuple[str, str]]


def unregistered_features() -> list[Feature[BattleContext]]:
    """Features only the experiments use: in no feature set, but under the same contract."""
    return [
        TeamEffect(),
        AtTurn("act_ttk_adv", ttk_advantage(CHART)),
    ]


def new_features() -> dict[str, Feature[BattleContext]]:
    """A new unfitted copy of every feature; one id can have several output selections."""
    found = list(unregistered_features())
    for name in sorted(feature_sets.FEATURE_SETS):
        features = feature_sets.build(name)
        first = features.base if isinstance(features, StagedFeatureSet) else features
        found.extend(first._entries[fid].feature for fid in first.feature_ids)
    keyed: dict[str, Feature[BattleContext]] = {}
    for f in found:
        key = f"{f.feature_id}:{'+'.join(f.output_names or ())}"
        if key in keyed and type(keyed[key]) is not type(f):
            key = f"{type(f).__module__}.{key}"  # v11_fs's own copy of a v11 feature
        keyed[key] = f
    return keyed


KEYS = sorted(new_features())
STAGED = sorted(
    n for n in feature_sets.FEATURE_SETS if isinstance(feature_sets.build(n), StagedFeatureSet)
)
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


def v11_switch_outputs() -> list[Row]:
    """v11's detector set (``ko_v11``, not in ``TRANSITION_SETS``) on every switch it scores."""
    feature = ko_v11.SwitchColumns()
    rows = []
    for record in read_records(FIXTURES):
        extractor = SwitchExtractor(pokedex().lookup(record, Missing()), CHART)
        for switch in [*extractor.examples(record), *extractor.candidates(record)]:
            rows.append([(k, float(v).hex()) for k, v in feature.compute(switch).items()])
    return rows


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


@cache
def staged_outputs(name: str) -> dict[str, list[Row]]:
    """Every table feature of a two-stage set, fitted on the fixtures, on its first-stage table."""
    features = feature_sets.build(name)
    data = battles()
    features.fit_transform(
        data.contexts, data.labels.tolist(), (), data.duplicate_groups().tolist()
    )
    table = features.base.to_frame(data.contexts)
    found: dict[str, list[Row]] = {}
    for f in features.stage:
        out = f.transform(table)
        assert out.dtype == np.float64 and out.shape == (len(table), len(f.output_names))
        found[f.feature_id] = [
            [(n, float(v).hex()) for n, v in zip(f.output_names, row, strict=True)] for row in out
        ]
    return found


@pytest.fixture(scope="module")
def outputs_by_hash_seed(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, list[Row]]]:
    """Every output under each hash seed, one process after the other: two at once share the
    few CPUs of a CI runner, where their LightGBM threads (``n_jobs=4`` each) spin against each
    other and the pair can take many times longer than both in sequence."""
    folder = tmp_path_factory.mktemp("hash_seeds")
    paths = [folder / f"{seed}.json" for seed in HASH_SEEDS]
    for seed, path in zip(HASH_SEEDS, paths, strict=True):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT)}
        command = [sys.executable, __file__, str(path)]
        subprocess.run(command, env=env, cwd=ROOT, check=True, timeout=WORKER_SECONDS + 30)
    return [json.loads(path.read_text()) for path in paths]


# the first of these tests also builds outputs_by_hash_seed (two full runs of every feature)
@pytest.mark.timeout(900)
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


def test_v11_detector_features_under_two_hash_seeds(
    outputs_by_hash_seed: list[dict[str, list[Row]]],
) -> None:
    first, second = (result["transitions:ko_v11"] for result in outputs_by_hash_seed)
    assert first == second and first


@pytest.mark.parametrize("name", STAGED)
def test_table_features_give_their_columns_as_floats_under_two_hash_seeds(
    name: str, outputs_by_hash_seed: list[dict[str, list[Row]]]
) -> None:
    found = staged_outputs(name)
    assert found and all(rows for rows in found.values())
    first, second = (result[f"staged:{name}"] for result in outputs_by_hash_seed)
    assert first == second == json.loads(json.dumps(found))


def test_contexts_never_carry_the_label() -> None:
    assert all("player_won" not in c.record for c in battles().contexts)


def test_every_family_of_features_is_checked() -> None:
    ids = {key.partition(":")[0] for key in KEYS}
    assert {"pool_bulk_spd", "team_rating", "team_encoding"} <= ids
    assert {"V11Static", "V11Bags", "V11Forecast"} <= ids
    assert STAGED == ["v11_fs"]
    assert "features.core.v11_fs.forecast.V11Forecast" in ids


if __name__ == "__main__":
    faulthandler.dump_traceback_later(WORKER_SECONDS, exit=True)
    all_outputs = {key: outputs(fitted(key)) for key in KEYS}
    all_outputs |= {f"transitions:{name}": transition_outputs(name) for name in TRANSITION_KEYS}
    all_outputs["transitions:ko_v11"] = v11_switch_outputs()
    all_outputs |= {f"staged:{name}": staged_outputs(name) for name in STAGED}
    Path(sys.argv[1]).write_text(json.dumps(all_outputs))
