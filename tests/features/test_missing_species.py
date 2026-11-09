"""Features on battles with a species without stats: NaN where stats are needed, no crash."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from data import Dataset
from domain.battle import BattleContext
from domain.pokedex import Pokedex, PokedexBuilder, species_without_stats
from features import sets as feature_sets
from features.core.matchup import active_duel, ttk_advantage
from features.core.moves import resisted_share
from features.core.pool import PoolBulk
from features.core.simulation import BattleSimulation
from features.core.team import HiddenTeamStrength, hpw_spd

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "battles.jsonl"
CHART = feature_sets.type_chart()


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return [json.loads(line) for line in FIXTURES.open()]


@pytest.fixture(scope="module")
def pokedex(records: list[dict]) -> Pokedex:
    return PokedexBuilder().add_records(records).build()


@pytest.fixture(scope="module")
def contexts(records: list[dict], pokedex: Pokedex) -> tuple[BattleContext, ...]:
    return Dataset.from_records(records, pokedex).contexts


@pytest.fixture(scope="module")
def stats_gone_on_active(contexts: tuple[BattleContext, ...]) -> BattleContext:
    """A battle whose final P2 active has no stats and is alive."""
    for context in contexts:
        active = context.side("p2").active.pokemon
        if not active.info.has_stats and not active.fainted:
            if not context.side("p1").active.pokemon.fainted:
                return context
    pytest.skip("no fixture battle ends with an alive P2 active without stats")


def test_fixtures_contain_species_without_stats(records: list[dict], pokedex: Pokedex) -> None:
    assert species_without_stats(records, pokedex)


@pytest.mark.parametrize("name", sorted(feature_sets.FEATURE_SETS))
def test_every_feature_set_survives(name: str, contexts: tuple[BattleContext, ...]) -> None:
    labels = [i % 2 for i in range(len(contexts))]
    features = feature_sets.build(name)
    features.fit(contexts, labels)
    table = features.to_frame(contexts)
    assert len(table) == len(contexts)


def test_active_matchups_are_nan(stats_gone_on_active: BattleContext) -> None:
    assert math.isnan(active_duel(CHART)(stats_gone_on_active))
    assert math.isnan(ttk_advantage(CHART)(stats_gone_on_active))


def test_simulation_is_nan(
    stats_gone_on_active: BattleContext, contexts: tuple[BattleContext, ...]
) -> None:
    simulation = BattleSimulation(CHART).fit(contexts)
    assert math.isnan(simulation.compute(stats_gone_on_active)["sim_seen_left"])


def test_bulk_share_is_nan_when_a_seen_species_is_alive(
    stats_gone_on_active: BattleContext,
) -> None:
    assert math.isnan(hpw_spd(stats_gone_on_active))


def test_resisted_share_is_nan_when_the_target_has_no_types(
    contexts: tuple[BattleContext, ...],
) -> None:
    values = []
    for context in contexts:
        hit = any(
            turn.p1.move is not None
            and context.species(turn.p1.target or turn.p2.active) is None
            and turn.p1.move.category != "status"
            for turn in context.history
        )
        if hit:
            values.append(resisted_share(CHART, context, "p1"))
    assert values and any(math.isnan(v) for v in values)


def test_hidden_guess_without_stats_gives_nan(contexts: tuple[BattleContext, ...]) -> None:
    feature = HiddenTeamStrength().fit(contexts)
    context = contexts[0]
    feature.guess = lambda revealed, size: ["no-stats-anywhere"]  # type: ignore[method-assign]
    assert math.isnan(feature.compute(context)["hidden_bst"])


def test_pool_bulk_with_a_likely_team_without_stats_gives_nan(
    contexts: tuple[BattleContext, ...],
) -> None:
    feature = PoolBulk().fit(contexts)
    context = next(c for c in contexts if c.side("p2").team_size > len(c.side("p2").team))
    feature.pool.most_likely = lambda revealed: ["no-stats-anywhere"]  # type: ignore[method-assign]
    assert math.isnan(feature.compute(context)["phpw_spd"])
