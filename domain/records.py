"""Battle records as they appear in the competition files.

:func:`read_records` yields validated records from a JSONL file, so the rest
of the code can trust their types. :func:`fingerprint` is the single
definition of "same battle".
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, cast

from domain.errors import InvalidRecordError

MoveCategory = Literal["physical", "special", "status"]

CATEGORIES = frozenset({"physical", "special", "status"})
STATUSES = frozenset({"nostatus", "par", "slp", "frz", "psn", "tox", "brn", "fnt"})
BOOSTED_STATS = frozenset({"atk", "def", "spa", "spd", "spe"})
BASE_STATS = ("hp", "atk", "def", "spa", "spd", "spe")
MAX_TEAM_SIZE = 6
MAX_STAGE = 6
LABEL_KEY = "player_won"
NOT_IDENTITY = ("battle_id", LABEL_KEY)


class PokemonDetails(TypedDict):
    """Species data of one Pokémon; only P1's team has an ``item``."""

    name: str
    level: int
    types: list[str]
    base_hp: int
    base_atk: int
    base_def: int
    base_spa: int
    base_spd: int
    base_spe: int
    item: NotRequired[str]


class MoveDetails(TypedDict):
    """A move used during a turn."""

    name: str
    type: str
    category: MoveCategory
    base_power: int
    accuracy: float
    priority: int


class PokemonState(TypedDict):
    """State of an active Pokémon at the end of a turn."""

    name: str
    hp_pct: float
    status: str  # "nostatus", "par", "slp", "frz", "psn", "tox", "brn", "fnt"
    effects: list[str]  # "noeffect" when empty
    boosts: dict[str, int]


class TurnRecord(TypedDict):
    """One row of the timeline."""

    turn: int
    p1_pokemon_state: PokemonState
    p1_move_details: MoveDetails | None
    p2_pokemon_state: PokemonState
    p2_move_details: MoveDetails | None


class BattleRecord(TypedDict):
    """One battle; the files store ``player_won`` as 0 / 1, and only in train."""

    battle_id: int
    p1_team_details: list[PokemonDetails]
    p2_lead_details: PokemonDetails
    battle_timeline: list[TurnRecord]
    player_won: NotRequired[int]


def read_records(path: str | Path) -> Iterator[BattleRecord]:
    """Yield the validated records of a JSONL file, skipping blank lines."""
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            where = f"{path}:{line_no}"
            try:
                record = validate(json.loads(line))
            except json.JSONDecodeError as exc:
                raise InvalidRecordError(f"{where}: invalid JSON", path=where) from exc
            except InvalidRecordError as exc:
                raise InvalidRecordError(
                    f"{where}: {exc}", battle_id=exc.battle_id, path=where
                ) from exc
            yield record


def validate(record: Mapping[str, Any]) -> BattleRecord:
    """Check keys and value ranges; the record is returned unchanged.

    Raises InvalidRecordError on the first problem found.
    """
    if not isinstance(record, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise InvalidRecordError("a record must be a JSON object")
    battle_id = record.get("battle_id")
    try:
        _check_record(record)
    except _Invalid as exc:
        raise InvalidRecordError(f"battle {battle_id}: {exc}", battle_id=battle_id) from exc
    return cast(BattleRecord, record)


def fingerprint(record: Mapping[str, Any]) -> str:
    """Hash of a battle's content, ignoring its id and label.

    It must be computed on the complete battle, not on a truncated copy.
    """
    content = {k: v for k, v in record.items() if k not in NOT_IDENTITY}
    return hashlib.sha1(json.dumps(content, sort_keys=True).encode()).hexdigest()


def without_label(record: Mapping[str, Any]) -> dict[str, Any]:
    """A shallow copy of ``record`` without the outcome: what features and repairs may read."""
    return {k: v for k, v in record.items() if k != LABEL_KEY}


class _Invalid(Exception):
    """A problem found by the checks below, before the battle id is added."""


def _check_record(record: Mapping[str, Any]) -> None:
    _require(_is_int(record.get("battle_id")), "battle_id: expected an integer")
    if "player_won" in record:
        _require(record["player_won"] in (0, 1), "player_won: expected 0 or 1")
    team = _list(_key(record, "p1_team_details", "record"), "p1_team_details")
    _require(1 <= len(team) <= MAX_TEAM_SIZE, f"p1_team_details: {len(team)} Pokémon")
    for i, details in enumerate(team):
        _check_details(details, f"p1_team_details[{i}]")
    _check_details(_key(record, "p2_lead_details", "record"), "p2_lead_details")
    timeline = _list(_key(record, "battle_timeline", "record"), "battle_timeline")
    for i, turn in enumerate(timeline):
        _check_turn(turn, f"battle_timeline[{i}]")


def _check_details(details: Any, where: str) -> None:
    fields = _mapping(details, where)
    _check_name(_key(fields, "name", where), f"{where}.name")
    _check_range(_key(fields, "level", where), 1, 100, f"{where}.level", integer=True)
    types = _list(_key(fields, "types", where), f"{where}.types")
    _require(
        1 <= len(types) <= 2 and all(isinstance(t, str) for t in types),
        f"{where}.types: expected one or two names",
    )
    for stat in BASE_STATS:
        key = f"base_{stat}"
        _check_range(_key(fields, key, where), 1, 255, f"{where}.{key}", integer=True)


def _check_turn(turn: Any, where: str) -> None:
    fields = _mapping(turn, where)
    _require(_is_int(_key(fields, "turn", where)), f"{where}.turn: expected an integer")
    for side in ("p1", "p2"):
        state = f"{side}_pokemon_state"
        _check_state(_key(fields, state, where), f"{where}.{state}")
        move = f"{side}_move_details"
        if _key(fields, move, where) is not None:
            _check_move(fields[move], f"{where}.{move}")


def _check_state(state: Any, where: str) -> None:
    fields = _mapping(state, where)
    _check_name(_key(fields, "name", where), f"{where}.name")
    _check_range(_key(fields, "hp_pct", where), 0.0, 1.0, f"{where}.hp_pct")
    _require(_key(fields, "status", where) in STATUSES, f"{where}.status: unknown status")
    effects = _list(_key(fields, "effects", where), f"{where}.effects")
    _require(all(isinstance(e, str) for e in effects), f"{where}.effects: expected names")
    boosts = _mapping(_key(fields, "boosts", where), f"{where}.boosts")
    for stat, stage in boosts.items():
        _require(stat in BOOSTED_STATS, f"{where}.boosts: unknown stat {stat!r}")
        _check_range(stage, -MAX_STAGE, MAX_STAGE, f"{where}.boosts.{stat}", integer=True)


def _check_move(move: Any, where: str) -> None:
    fields = _mapping(move, where)
    _check_name(_key(fields, "name", where), f"{where}.name")
    _check_name(_key(fields, "type", where), f"{where}.type")
    _require(_key(fields, "category", where) in CATEGORIES, f"{where}.category: unknown")
    _require(_is_int(_key(fields, "base_power", where)), f"{where}.base_power: expected an integer")
    _require(fields["base_power"] >= 0, f"{where}.base_power: negative")
    _check_range(_key(fields, "accuracy", where), 0.0, 1.0, f"{where}.accuracy")
    _require(_is_int(_key(fields, "priority", where)), f"{where}.priority: expected an integer")


def _check_name(value: Any, where: str) -> None:
    _require(isinstance(value, str) and bool(value), f"{where}: expected a name")


def _check_range(value: Any, low: float, high: float, where: str, integer: bool = False) -> None:
    number = _is_int(value) or (not integer and isinstance(value, float))
    _require(number and low <= value <= high, f"{where}: {value!r} not in {low}..{high}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _key(fields: Mapping[str, Any], key: str, where: str) -> Any:
    _require(key in fields, f"{where}: missing key {key!r}")
    return fields[key]


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{where}: expected an object")
    return cast(Mapping[str, Any], value)


def _list(value: Any, where: str) -> list[Any]:
    _require(isinstance(value, list), f"{where}: expected a list")
    return cast(list[Any], value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _Invalid(message)
