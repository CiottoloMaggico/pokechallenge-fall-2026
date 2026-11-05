"""Gen 2 field and end-of-turn rules behind seven transition columns.

Split from :mod:`domain.transitions`, which describes a switch and adds these
values to it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from domain.damage import Combatant, DamageCalculator, Field
from domain.moves import Move
from domain.pokedex import PokemonGen2, SpeciesLookup

FAINTED = "fnt"
MAX_STAGE = 6
MISSING = float("nan")
SIDES = ("p1", "p2")
SCREEN_MOVES = {"reflect": "reflect", "lightscreen": "lightscreen"}
WEATHER_MOVES = {"raindance": "rain", "sunnyday": "sun", "sandstorm": "sand"}
FIELD_TURNS = 5
PARTIAL_TRAPS = frozenset({"wrap", "bind", "clamp", "firespin", "whirlpool"})
SAND_IMMUNE = frozenset({"rock", "ground", "steel"})
LEFTOVERS_SHARE = 0.95  # share of known Pokémon holding Leftovers in the records

Turn = Mapping[str, Any]
FieldState = tuple[str | None, Mapping[str, frozenset[str]]]


def stages(boosts: Mapping[str, int]) -> dict[str, int]:
    """Boosts clamped to the legal stages."""
    return {k: max(-MAX_STAGE, min(MAX_STAGE, v)) for k, v in boosts.items()}


def attacker_status(status: str) -> str:
    """A Pokémon shown as fainted attacked before fainting: without a status."""
    return "nostatus" if status == FAINTED else status


def _fresh_move(timeline: Sequence[Turn], side: str, i: int) -> Turn | None:
    """``side``'s move really used at row ``i`` (the records repeat stale ones)."""
    move = timeline[i][f"{side}_move_details"]
    if move is None or i == 0:
        return move
    before = timeline[i - 1]
    if before[f"{side}_pokemon_state"]["name"] != timeline[i][f"{side}_pokemon_state"]["name"]:
        return None
    previous = before[f"{side}_move_details"]
    return None if previous is not None and previous["name"] == move["name"] else move


def field_tracker(timeline: Sequence[Turn]) -> list[FieldState]:
    """Weather and each side's screens after every row (Gen 2: 5 turns)."""
    weather: str | None = None
    weather_left = 0
    screens: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
    result: list[FieldState] = []
    for i in range(len(timeline)):
        for side in SIDES:
            move = _fresh_move(timeline, side, i)
            if move is None:
                continue
            if move["name"] in WEATHER_MOVES:
                weather, weather_left = WEATHER_MOVES[move["name"]], FIELD_TURNS
            if move["name"] in SCREEN_MOVES:
                screens[side][SCREEN_MOVES[move["name"]]] = FIELD_TURNS
        active = {s: frozenset(k for k, v in d.items() if v > 0) for s, d in screens.items()}
        result.append((weather if weather_left > 0 else None, active))
        weather_left = max(0, weather_left - 1)
        for left in screens.values():
            for key in left:
                left[key] -= 1
    return result


def status_counters(timeline: Sequence[Turn], side: str) -> tuple[list[int], list[int]]:
    """Rows the active Pokémon has been badly poisoned / under Perish Song."""
    tox: list[int] = []
    perish: list[int] = []
    t = p = 0
    for i, row in enumerate(timeline):
        state = row[f"{side}_pokemon_state"]
        same = i > 0 and timeline[i - 1][f"{side}_pokemon_state"]["name"] == state["name"]
        t = (t + 1 if same else 1) if state["status"] == "tox" else 0
        p = (p + 1 if same else 1) if "perish" in state["effects"] else 0
        tox.append(t)
        perish.append(p)
    return tox, perish


def mechanics(
    timeline: Sequence[Turn],
    side: str,
    before: int,
    index: int,
    species: SpeciesLookup,
    damage: DamageCalculator,
    field_state: FieldState,
    tox_turns: int,
    perish_turns: int,
) -> dict[str, float]:
    """Gen 2 mechanics the base features ignore.

    * ``a_dmg_field``: expected damage of the opposing move on the leaver
      with the weather and the leaver's Reflect / Light Screen;
    * ``x_residual``: end-of-turn damage on the leaver (poison and burn
      1/8, Toxic n/16, Leech Seed 1/8, Curse 1/4, Nightmare 1/4, partial
      trapping 1/16, sandstorm 1/8 unless Rock / Ground / Steel) minus the
      expected Leftovers recovery;
    * ``x_hp_after``: leaver's HP minus both;
    * ``a_priority``: priority of the opposing move;
    * ``x_perish_turns`` / ``x_tox_turns``: rows under Perish Song / Toxic;
    * ``a_drop_any``: largest HP loss of any opposing Pokémon active over
      the rows of the change, against its last known HP. It sees the hit
      taken by an opponent that switched in on that turn (an Explosion,
      the leaver's last attack), which ``a_drop`` misses.
    """
    other = "p2" if side == "p1" else "p1"
    weather, screens = field_state
    leaver_state = timeline[before][f"{side}_pokemon_state"]
    attacker_state = timeline[index][f"{other}_pokemon_state"]
    move = timeline[index][f"{other}_move_details"]
    leaver = species(leaver_state["name"])
    attacker = species(attacker_state["name"])
    residual = _residual(leaver_state, leaver, weather, tox_turns)
    field = Field.of(weather, screens[side])
    expected = _field_damage(move, attacker_state, attacker, leaver_state, leaver, field, damage)
    return {
        "a_dmg_field": expected,
        "x_residual": residual,
        "x_hp_after": leaver_state["hp_pct"] - expected - residual,
        "a_priority": float(move["priority"]) if move else 0.0,
        "x_perish_turns": float(perish_turns),
        "x_tox_turns": float(tox_turns),
        "a_drop_any": _drop_any(timeline, other, before, index),
    }


def _residual(
    state: Mapping[str, Any], leaver: PokemonGen2 | None, weather: str | None, tox_turns: int
) -> float:
    status, effects = state["status"], set(state["effects"])
    residual = 0.0
    if status in ("psn", "brn"):
        residual += 1 / 8
    if status == "tox":
        residual += min(15, tox_turns + 1) / 16
    if "leechseed" in effects:
        residual += 1 / 8
    if "curse" in effects:
        residual += 1 / 4
    if "nightmare" in effects and status == "slp":
        residual += 1 / 4
    if effects & PARTIAL_TRAPS:
        residual += 1 / 16
    if weather == "sand" and leaver is not None and not set(leaver.types) & SAND_IMMUNE:
        residual += 1 / 8
    if weather == "sand" and leaver is None:
        residual = MISSING
    return residual - LEFTOVERS_SHARE / 16


def _field_damage(
    move: Turn | None,
    attacker_state: Mapping[str, Any],
    attacker: PokemonGen2 | None,
    leaver_state: Mapping[str, Any],
    leaver: PokemonGen2 | None,
    field: Field,
    damage: DamageCalculator,
) -> float:
    if move is None or move["category"] == "status":
        return 0.0
    if leaver is None or attacker is None:
        return MISSING
    a = Combatant(
        attacker,
        boosts=stages(attacker_state.get("boosts") or {}),
        status=attacker_status(attacker_state["status"]),
    )
    d = Combatant(leaver, boosts=stages(leaver_state.get("boosts") or {}))
    mean = damage.expected(a, d, Move.from_details(move), field, with_accuracy=True)
    return 0.0 if mean is None else mean / d.max_hp


def _drop_any(timeline: Sequence[Turn], other: str, before: int, index: int) -> float:
    drop_any = 0.0
    for r in range(before + 1, index + 1):
        state = timeline[r][f"{other}_pokemon_state"]
        last = 1.0
        for q in range(r - 1, -1, -1):
            earlier = timeline[q][f"{other}_pokemon_state"]
            if earlier["name"] == state["name"]:
                last = earlier["hp_pct"]
                break
        drop_any = max(drop_any, last - state["hp_pct"])
    return drop_any
