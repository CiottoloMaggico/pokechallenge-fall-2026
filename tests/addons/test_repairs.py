"""The repairs of ``addons`` behind ``TimelineRepair``, on the fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from addons import check_spec, load_repair
from addons.knockouts_model import ArtifactError, ModelKnockoutRepair
from addons.knockouts_rule import RuleKnockoutRepair
from data import Dataset
from detector import DetectorTask, load_task
from domain.damage import DamageCalculator
from domain.pokedex import Pokedex, PokedexBuilder
from domain.records import read_records, without_label
from domain.repair import NO_REPAIR, TimelineRepair
from features import transitions
from features.common import type_chart
from features.core.knockouts import expected_hidden_knockouts, unsure_hidden_knockouts
from features.core.pool import soft_material_hp
from gym import Gym
from models import build

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "battles.jsonl"
TEAM_SIZE = 6
THRESHOLD = 0.5
INFO: dict[str, Any] = {
    "version": "test",
    "feature_set": "ko_domain30",
    "model": "hgb_ko",
    "params": {"max_iter": 30},
    "threshold": THRESHOLD,
    "train_sha256": "fixtures",
    "pokedex_sha256": None,
}


@pytest.fixture(scope="module")
def pokedex() -> Pokedex:
    return PokedexBuilder().add_records(read_records(FIXTURES)).build()


@pytest.fixture(scope="module")
def task(pokedex: Pokedex) -> DetectorTask:
    return load_task(FIXTURES, pokedex)


@pytest.fixture(scope="module")
def repair(pokedex: Pokedex, task: DetectorTask) -> ModelKnockoutRepair:
    gym = Gym([build("hgb_ko", max_iter=30)], [transitions.build("ko_domain30")])
    (run,) = gym.train(task.p1)
    return ModelKnockoutRepair(run, INFO, pokedex, DamageCalculator(type_chart()))


@pytest.fixture(scope="module")
def dataset(pokedex: Pokedex, repair: ModelKnockoutRepair) -> Dataset:
    return Dataset.from_jsonl(FIXTURES, pokedex, repair=repair)


def test_implements_the_protocol(repair: ModelKnockoutRepair) -> None:
    implementation: TimelineRepair = repair
    described = json.loads(json.dumps(implementation.describe()))
    assert described == {
        "name": "knockouts_model",
        "version": "test",
        "params": {k: v for k, v in INFO.items() if k != "version"},
    }


def test_repairs_are_the_probabilities_over_the_threshold(
    repair: ModelKnockoutRepair, dataset: Dataset
) -> None:
    repaired = 0
    for context in dataset:
        probabilities = repair.probabilities(context.record)
        timeline = context.record["battle_timeline"]
        expected = {
            i: timeline[i - 1]["p2_pokemon_state"]["name"]
            for i, p in probabilities.items()
            if p >= THRESHOLD
        }
        assert repair.repairs(context.record) == expected
        repaired += len(expected)
    assert repaired > 0


def test_the_batch_of_the_checks_scores_like_the_repair(
    repair: ModelKnockoutRepair, task: DetectorTask
) -> None:
    by_battle = [
        p
        for record in read_records(FIXTURES)
        for _, p in sorted(repair.probabilities(record).items())
    ]
    assert np.array_equal(repair._run.predict_proba(task.p2), by_battle)


def test_the_label_does_not_change_the_scores(repair: ModelKnockoutRepair) -> None:
    record = next(read_records(FIXTURES))
    assert repair.probabilities(record) == repair.probabilities(without_label(record))


def test_repaired_faints_reach_the_replay(repair: ModelKnockoutRepair, dataset: Dataset) -> None:
    context = next(c for c in dataset if repair.repairs(c.record))
    for name in repair.repairs(context.record).values():
        assert context.side("p2").team[name].fainted


def test_unsure_is_the_sum_below_the_threshold(
    repair: ModelKnockoutRepair, dataset: Dataset
) -> None:
    for context in dataset:
        probabilities = repair.probabilities(context.record).values()
        below = [p for p in probabilities if p < THRESHOLD]
        assert unsure_hidden_knockouts(context) == min(1.0, sum(below) / TEAM_SIZE)
        assert expected_hidden_knockouts(context) == min(1.0, sum(probabilities) / TEAM_SIZE)


def test_truncated_contexts_score_only_their_own_rows(
    repair: ModelKnockoutRepair, dataset: Dataset
) -> None:
    context = next(c for c in dataset if repair.probabilities(c.record))
    last_scored = max(repair.probabilities(context.record))
    early = context.at_turn(last_scored)
    assert all(i < last_scored for i in repair.probabilities(early.record))


def test_without_a_repair_the_detector_features_see_no_knockouts(pokedex: Pokedex) -> None:
    context = Dataset.from_jsonl(FIXTURES, pokedex)[0]
    assert expected_hidden_knockouts(context) == 0
    assert unsure_hidden_knockouts(context) == 0
    assert isinstance(soft_material_hp(context), float)


def test_the_rule_gives_zero_or_one_and_repairs_the_ones(pokedex: Pokedex) -> None:
    rule = RuleKnockoutRepair(pokedex, DamageCalculator(type_chart()))
    scored = repaired = 0
    for record in read_records(FIXTURES):
        probabilities = rule.probabilities(record)
        assert set(probabilities.values()) <= {0.0, 1.0}
        assert set(rule.repairs(record)) == {i for i, p in probabilities.items() if p == 1.0}
        scored += len(probabilities)
        repaired += len(rule.repairs(record))
    assert 0 < repaired < scored
    assert json.loads(json.dumps(rule.describe()))["name"] == "knockouts_rule"


@pytest.mark.parametrize("spec", ["none", "rule", "model=results/addons/knockouts/v3"])
def test_valid_specs(spec: str) -> None:
    assert check_spec(spec) == spec


@pytest.mark.parametrize("spec", ["", "model", "model=", "rule=1", "detector"])
def test_invalid_specs(spec: str) -> None:
    with pytest.raises(ValueError, match="unknown repair"):
        check_spec(spec)


def test_load_repair_by_name(pokedex: Pokedex, tmp_path: Path) -> None:
    assert load_repair("none", pokedex, type_chart()) is NO_REPAIR
    assert isinstance(load_repair("rule", pokedex, type_chart()), RuleKnockoutRepair)
    with pytest.raises(ArtifactError, match="detector train --version v9"):
        load_repair(f"model={tmp_path / 'v9'}", pokedex, type_chart())
