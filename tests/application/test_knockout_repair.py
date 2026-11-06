"""The current detector behind ``TimelineRepair``, on the fixtures."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest

from data import Dataset
from domain.pokedex import Pokedex, PokedexBuilder
from domain.records import read_records, without_label
from domain.repair import TimelineRepair
from domain.typechart import TypeChart
from features.core.knockouts import expected_hidden_knockouts, unsure_hidden_knockouts
from features.core.pool import soft_material_hp
from knockout_detector import (
    DEFAULT_THRESHOLD,
    DetectorRepair,
    KnockoutDetector,
    detector_from_files,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "battles.jsonl"
TEAM_SIZE = 6


@pytest.fixture(scope="module")
def pokedex() -> Pokedex:
    return PokedexBuilder().add_records(read_records(FIXTURES)).build()


@pytest.fixture(scope="module")
def chart() -> TypeChart:
    return TypeChart.from_csv(ROOT / "assets" / "type_chart_gen2.csv")


@pytest.fixture(scope="module")
def repair(pokedex: Pokedex, chart: TypeChart) -> DetectorRepair:
    return detector_from_files(pokedex, chart, (FIXTURES,))


@pytest.fixture(scope="module")
def dataset(pokedex: Pokedex, repair: DetectorRepair) -> Dataset:
    return Dataset.from_jsonl(FIXTURES, pokedex, repair=repair)


def test_implements_the_protocol(repair: DetectorRepair) -> None:
    implementation: TimelineRepair = repair
    described = json.loads(json.dumps(implementation.describe()))
    assert described["name"] == "knockout_detector"
    assert described["params"]["threshold"] == DEFAULT_THRESHOLD


def test_repairs_are_the_probabilities_over_the_threshold(
    repair: DetectorRepair, dataset: Dataset
) -> None:
    repaired = 0
    for context in dataset:
        probabilities = repair.probabilities(context.record)
        timeline = context.record["battle_timeline"]
        expected = {
            i: timeline[i - 1]["p2_pokemon_state"]["name"]
            for i, p in probabilities.items()
            if p >= DEFAULT_THRESHOLD
        }
        assert repair.repairs(context.record) == expected
        repaired += len(expected)
    assert repaired > 0


def test_the_label_does_not_change_the_scores(pokedex: Pokedex, chart: TypeChart) -> None:
    repair = detector_from_files(pokedex, chart, (FIXTURES,))
    record = next(read_records(FIXTURES))
    assert repair.probabilities(record) == repair.probabilities(without_label(record))


def test_the_detector_is_fitted_without_labels(
    pokedex: Pokedex, chart: TypeChart, monkeypatch: pytest.MonkeyPatch
) -> None:
    fitted: list[Mapping[str, Any]] = []
    original = KnockoutDetector.fit

    def spy(self: KnockoutDetector, records: Iterable[Mapping[str, Any]]) -> KnockoutDetector:
        fitted.extend(records)
        return original(self, fitted)

    monkeypatch.setattr(KnockoutDetector, "fit", spy)
    detector_from_files(pokedex, chart, (FIXTURES,))
    assert fitted
    assert all("player_won" not in record for record in fitted)


def test_repaired_faints_reach_the_replay(repair: DetectorRepair, dataset: Dataset) -> None:
    context = next(c for c in dataset if repair.repairs(c.record))
    for name in repair.repairs(context.record).values():
        assert context.side("p2").team[name].fainted


def test_unsure_is_the_sum_below_the_threshold(repair: DetectorRepair, dataset: Dataset) -> None:
    for context in dataset:
        probabilities = repair.probabilities(context.record).values()
        below = [p for p in probabilities if p < DEFAULT_THRESHOLD]
        assert unsure_hidden_knockouts(context) == min(1.0, sum(below) / TEAM_SIZE)
        assert expected_hidden_knockouts(context) == min(1.0, sum(probabilities) / TEAM_SIZE)


def test_truncated_contexts_score_only_their_own_rows(
    repair: DetectorRepair, dataset: Dataset
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
