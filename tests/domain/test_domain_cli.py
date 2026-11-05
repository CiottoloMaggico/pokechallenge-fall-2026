"""`python -m domain replay | check`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from domain.__main__ import main

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "battles.jsonl"


def first_record() -> dict[str, Any]:
    with FIXTURES.open() as fh:
        return json.loads(fh.readline())


def write_records(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_replay_prints_one_line_per_turn(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = first_record()
    species = tmp_path / "species.json"
    assert main(["build-pokedex", str(FIXTURES), "--out", str(species)]) == 0
    capsys.readouterr()
    args = ["replay", str(FIXTURES), "--battle", str(record["battle_id"])]
    assert main([*args, "--pokedex", str(species)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(record["battle_timeline"])
    assert lines[0].startswith("turn  1  p1 ")
    assert record["battle_timeline"][0]["p2_pokemon_state"]["name"] in lines[0]


def test_replay_of_an_absent_battle_is_a_domain_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    species = tmp_path / "species.json"
    assert main(["build-pokedex", str(FIXTURES), "--out", str(species)]) == 0
    assert main(["replay", str(FIXTURES), "--battle", "-1", "--pokedex", str(species)]) == 2
    assert "battle -1 is not in" in capsys.readouterr().err


def test_check_builds_the_pokedex_only_from_the_first_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = first_record()
    train = write_records(tmp_path / "train.jsonl", [record])
    other = json.loads(json.dumps(record))
    other["battle_timeline"][-1]["p2_pokemon_state"]["name"] = "hidden"
    test = write_records(tmp_path / "test.jsonl", [other])
    assert main(["check", str(train), str(test)]) == 0
    pokedex, train_line, test_line = capsys.readouterr().out.splitlines()
    assert pokedex.startswith(f"pokedex of {train}: ")
    assert train_line.startswith(f"{train}: 1 valid battles")
    assert "hidden" not in train_line
    assert test_line.startswith(f"{test}: 1 valid battles")
    assert "'hidden': 1" in test_line


def test_check_reports_an_invalid_row_with_exit_code_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = first_record()
    train = write_records(tmp_path / "train.jsonl", [record])
    broken = write_records(tmp_path / "test.jsonl", [{**record, "player_won": 7}])
    assert main(["check", str(train), str(broken)]) == 2
    err = capsys.readouterr().err
    assert "test.jsonl:1" in err
    assert len(err.splitlines()) == 1
