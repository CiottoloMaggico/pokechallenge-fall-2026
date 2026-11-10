"""v11's group helpers on the fixtures: the view of a battle, stable team ids, TeamModel."""

from __future__ import annotations

from pathlib import Path

import pytest

from addons import load_repair
from data import Dataset, load_records
from domain.battle import BattleContext
from domain.pokedex import PokedexBuilder
from features import sets as feature_sets
from features.core.v11.battle import Battle, battle_of
from features.core.v11.state import tracks_of
from features.core.v11.team import team_id
from features.core.v11.team_model import TeamModel

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "battles.jsonl"


@pytest.fixture(scope="module")
def contexts() -> tuple[BattleContext, ...]:
    records = load_records(FIXTURES)
    pokedex = PokedexBuilder().add_records(records).build()
    repair = load_repair("rule", pokedex, feature_sets.type_chart())
    return Dataset.from_records(records, pokedex, repair=repair).contexts


def test_the_view_reads_the_record_and_the_repair(contexts: tuple[BattleContext, ...]) -> None:
    for context in contexts[:30]:
        view = battle_of(context)
        assert battle_of(context) is view and tracks_of(context) is tracks_of(context)
        assert view.p2_faint == context.repair.probabilities(context.record)
        assert view.n == len(context.record["battle_timeline"])
        assert view.stats(view.lead2["name"]) == context.species(view.lead2["name"])


def test_a_repaired_faint_counts_in_the_p2_track(contexts: tuple[BattleContext, ...]) -> None:
    repaired = 0
    for context in contexts:
        view, (_, t2) = battle_of(context), tracks_of(context)
        for i, p in view.p2_faint.items():
            if p == 1.0:
                assert t2["pf"][view.sides["p2"].names[i - 1]] == 1.0
                repaired += 1
    assert repaired > 0


def test_team_id_ignores_order_and_is_stable() -> None:
    names = ["snorlax", "zapdos", "cloyster"]
    assert team_id(names) == team_id(reversed(names))
    # first 12 hex digits of md5("cloyster|snorlax|zapdos"): no hash seed involved
    assert team_id(names) == float(0x60C632724A64) == 106404366142052.0
    assert team_id(names) != team_id(names[:2])


def test_team_model_fills_the_hidden_slots(contexts: tuple[BattleContext, ...]) -> None:
    views: list[Battle] = [battle_of(c) for c in contexts]
    team_model = TeamModel(views)
    for b in views[:20]:
        revealed = list(b.sides["p2"].revealed)
        n_hidden = 6 - len(revealed)
        expected = team_model.expected_unrevealed(revealed, n_hidden)
        if n_hidden == 0:
            assert expected == {}
            continue
        assert not set(expected) & set(revealed)
        assert all(0.0 < p <= 1.0 for p in expected.values())
        assert sum(expected.values()) == pytest.approx(n_hidden, abs=0.05)
