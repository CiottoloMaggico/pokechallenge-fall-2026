"""v11's feature groups on the fixtures: same keys for every battle, stable team ids, TeamModel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from features.core.v11.bags import Vocab
from features.core.v11.build import battle_features
from features.core.v11.faint import fit_detector
from features.core.v11.mechanics import Pokedex
from features.core.v11.replay import Battle, parse
from features.core.v11.simulation import MoveTable
from features.core.v11.switching import SwitchRates
from features.core.v11.team import team_id
from features.core.v11.team_model import TeamModel

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "battles.jsonl"
N_TRAIN = 150
PREFIXES = ("s_", "f_", "t_", "a_", "e_", "m_", "w_", "x_", "bs_", "bh_", "bm_", "k_", "d_")
PREFIXES += ("v_", "z_", "c_")

Record = dict[str, Any]
Fitted = tuple[list[Battle], list[Battle], Vocab, TeamModel, MoveTable, SwitchRates]


@pytest.fixture(scope="module")
def fitted() -> Fitted:
    records = [json.loads(line) for line in FIXTURES.open()]
    dex = Pokedex(records[:N_TRAIN])
    train, test = parse(records[:N_TRAIN], dex), parse(records[N_TRAIN:], dex)
    fit_detector(train, test)
    return train, test, Vocab(train), TeamModel(train), MoveTable(train), SwitchRates(train)


def test_every_battle_gets_the_same_keys(fitted: Fitted) -> None:
    train, test, vocab, team_model, moves, rates = fitted
    rows = [battle_features(b, vocab, team_model, moves, rates) for b in train + test]
    keys = list(rows[0])
    assert all(list(row) == keys for row in rows)
    assert keys[0] == "battle_id"
    assert all(k.startswith(PREFIXES) for k in keys[1:])
    assert {k.split("_")[0] for k in keys[1:]} == {p.rstrip("_") for p in PREFIXES}


def test_team_id_ignores_order_and_is_stable() -> None:
    names = ["snorlax", "zapdos", "cloyster"]
    assert team_id(names) == team_id(reversed(names))
    # first 12 hex digits of md5("cloyster|snorlax|zapdos"): no hash seed involved
    assert team_id(names) == float(0x60C632724A64) == 106404366142052.0
    assert team_id(names) != team_id(names[:2])


def test_team_model_fills_the_hidden_slots(fitted: Fitted) -> None:
    train, _, _, team_model, _, _ = fitted
    for b in train[:20]:
        revealed = list(b.sides["p2"].revealed)
        n_hidden = 6 - len(revealed)
        expected = team_model.expected_unrevealed(revealed, n_hidden)
        if n_hidden == 0:
            assert expected == {}
            continue
        assert not set(expected) & set(revealed)
        assert all(0.0 < p <= 1.0 for p in expected.values())
        assert sum(expected.values()) == pytest.approx(n_hidden, abs=0.05)
