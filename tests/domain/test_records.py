"""Record validation, reading and the battle fingerprint."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from domain.errors import DomainError, InvalidRecordError
from domain.moves import Move
from domain.records import fingerprint, read_records, validate


def details(name: str = "snorlax") -> dict[str, Any]:
    return {
        "name": name,
        "level": 100,
        "types": ["normal", "notype"],
        "base_hp": 160,
        "base_atk": 110,
        "base_def": 65,
        "base_spa": 65,
        "base_spd": 110,
        "base_spe": 30,
    }


def state(name: str = "snorlax") -> dict[str, Any]:
    return {
        "name": name,
        "hp_pct": 0.5,
        "status": "nostatus",
        "effects": ["noeffect"],
        "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
    }


def move_details() -> dict[str, Any]:
    return {
        "name": "bodyslam",
        "type": "normal",
        "category": "physical",
        "base_power": 85,
        "accuracy": 1.0,
        "priority": 0,
    }


def record() -> dict[str, Any]:
    return {
        "battle_id": 7,
        "player_won": 1,
        "p1_team_details": [{**details(), "item": "leftovers"}],
        "p2_lead_details": details("zapdos"),
        "battle_timeline": [
            {
                "turn": 1,
                "p1_pokemon_state": state(),
                "p1_move_details": move_details(),
                "p2_pokemon_state": state("zapdos"),
                "p2_move_details": None,
            }
        ],
    }


def test_a_valid_record_is_returned_unchanged() -> None:
    raw = record()
    before = copy.deepcopy(raw)
    assert validate(raw) is raw
    assert raw == before


def test_the_label_is_optional() -> None:
    raw = record()
    del raw["player_won"]
    assert validate(raw) is raw


def _delete(raw: dict[str, Any], *path: str | int) -> None:
    target: Any = raw
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]


def _setter(value: Any, *path: str | int) -> Callable[[dict[str, Any]], None]:
    def apply(raw: dict[str, Any]) -> None:
        target: Any = raw
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return apply


FIRST_TURN = ("battle_timeline", 0)
P1_STATE = (*FIRST_TURN, "p1_pokemon_state")
P1_MOVE = (*FIRST_TURN, "p1_move_details")

INVALID = {
    "battle_id_type": _setter("seven", "battle_id"),
    "battle_id_bool": _setter(True, "battle_id"),
    "label_range": _setter(2, "player_won"),
    "empty_team": _setter([], "p1_team_details"),
    "team_too_large": _setter([details()] * 7, "p1_team_details"),
    "team_not_a_list": _setter({}, "p1_team_details"),
    "level_range": _setter(101, "p2_lead_details", "level"),
    "level_type": _setter("100", "p2_lead_details", "level"),
    "empty_name": _setter("", "p2_lead_details", "name"),
    "too_many_types": _setter(["a", "b", "c"], "p2_lead_details", "types"),
    "base_stat_range": _setter(0, "p2_lead_details", "base_atk"),
    "timeline_not_a_list": _setter({}, "battle_timeline"),
    "turn_type": _setter(1.5, *FIRST_TURN, "turn"),
    "hp_range": _setter(1.5, *P1_STATE, "hp_pct"),
    "hp_nan": _setter(float("nan"), *P1_STATE, "hp_pct"),
    "status": _setter("asleep", *P1_STATE, "status"),
    "effects_type": _setter("noeffect", *P1_STATE, "effects"),
    "boost_range": _setter(7, *P1_STATE, "boosts", "atk"),
    "boost_stat": _setter(1, *P1_STATE, "boosts", "hp"),
    "category": _setter("other", *P1_MOVE, "category"),
    "base_power": _setter(-1, *P1_MOVE, "base_power"),
    "accuracy": _setter(1.5, *P1_MOVE, "accuracy"),
    "priority": _setter("0", *P1_MOVE, "priority"),
}


@pytest.mark.parametrize("case", INVALID)
def test_invalid_values(case: str) -> None:
    raw = record()
    INVALID[case](raw)
    with pytest.raises(InvalidRecordError) as caught:
        validate(raw)
    assert caught.value.path is None
    assert isinstance(caught.value, DomainError)


@pytest.mark.parametrize(
    "path",
    [
        ("battle_id",),
        ("p1_team_details",),
        ("p2_lead_details",),
        ("battle_timeline",),
        ("p2_lead_details", "base_spe"),
        (*FIRST_TURN, "turn"),
        (*FIRST_TURN, "p2_move_details"),
        (*P1_STATE, "boosts"),
        (*P1_MOVE, "priority"),
    ],
)
def test_missing_keys(path: tuple[str | int, ...]) -> None:
    raw = record()
    _delete(raw, *path)
    with pytest.raises(InvalidRecordError, match="battle_id|missing key"):
        validate(raw)


def test_the_error_names_the_battle_and_the_field() -> None:
    raw = record()
    _setter(1.5, *P1_STATE, "hp_pct")(raw)
    with pytest.raises(InvalidRecordError, match=r"battle 7: .*p1_pokemon_state\.hp_pct") as caught:
        validate(raw)
    assert caught.value.battle_id == 7


def test_a_record_must_be_an_object() -> None:
    with pytest.raises(InvalidRecordError, match="JSON object"):
        validate([])  # type: ignore[arg-type]


def test_read_records_skips_blank_lines_and_keeps_the_order(tmp_path: Path) -> None:
    second = {**record(), "battle_id": 8}
    path = tmp_path / "battles.jsonl"
    path.write_text(f"{json.dumps(record())}\n\n{json.dumps(second)}\n", encoding="utf-8")
    assert [r["battle_id"] for r in read_records(path)] == [7, 8]


def test_read_records_reports_invalid_json_with_its_line(tmp_path: Path) -> None:
    path = tmp_path / "battles.jsonl"
    path.write_text(f"{json.dumps(record())}\n{{oops\n", encoding="utf-8")
    with pytest.raises(InvalidRecordError, match="invalid JSON") as caught:
        list(read_records(path))
    assert caught.value.path == f"{path}:2"


def test_read_records_reports_invalid_records_with_their_line(tmp_path: Path) -> None:
    raw = record()
    raw["player_won"] = 2
    path = tmp_path / "battles.jsonl"
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(InvalidRecordError, match="player_won") as caught:
        list(read_records(path))
    assert caught.value.path == f"{path}:1"
    assert caught.value.battle_id == 7


def test_fingerprint_ignores_id_and_label_only() -> None:
    base = fingerprint(record())
    assert fingerprint({**record(), "battle_id": 99}) == base
    assert fingerprint({k: v for k, v in record().items() if k != "player_won"}) == base
    changed = record()
    changed["battle_timeline"][0]["turn"] = 2
    assert fingerprint(changed) != base


def test_fingerprint_does_not_depend_on_the_key_order() -> None:
    raw = record()
    assert fingerprint(dict(reversed(list(raw.items())))) == fingerprint(raw)


def test_fingerprint_is_the_sha1_of_the_sorted_json() -> None:
    raw = record()
    content = {k: v for k, v in raw.items() if k not in ("battle_id", "player_won")}
    expected = hashlib.sha1(json.dumps(content, sort_keys=True).encode()).hexdigest()
    assert fingerprint(raw) == expected


def test_move_from_details() -> None:
    assert Move.from_details(move_details()) == Move("bodyslam", "normal", "physical", 85, 1.0, 0)
