"""v11's second simulation (``z_`` columns): typical moves and a STAB move per type always
available, turns still lost asleep or frozen, weather and screens in the duel of the actives, and
a variant against the unrevealed Player 2 Pokémon guessed from team co-occurrences.

A copy of the second half of the original's ``features/simulation.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from domain import expected_damage as ed
from domain.pokedex import PokemonGen2
from domain.timeline import Move
from features.core.v11_fs import moves as mx
from features.core.v11_fs.battle import Battle
from features.core.v11_fs.matchup import Fighter, proxies
from features.core.v11_fs.moves import CHART
from features.core.v11_fs.simulation import MAX_DUELS, MAX_HITS, MIN_HP, Cache, MoveTable, is_attack
from features.core.v11_fs.state import Track
from features.core.v11_fs.team_model import TeamModel

FIELD_TURNS = 5
WEATHER_BOOST: dict[str | None, dict[str, float]] = {
    "raindance": {"water": 1.5, "fire": 0.5},
    "sunnyday": {"fire": 1.5, "water": 0.5},
}
SCREEN_OF = {"physical": "reflect", "special": "lightscreen"}

Field = tuple[str | None, dict[str, set[str]]]


def _field(b: Battle) -> Field:
    """Weather and screens still up at the last entry."""
    weather: tuple[str | None, int] = (None, -FIELD_TURNS)
    screens: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
    for i in range(b.n):
        for s in ("p1", "p2"):
            m = b.timeline.used_move(s, i)
            if m is not None and m["name"] in mx.WEATHER:
                weather = (m["name"], i)
            elif m is not None and m["name"] in mx.SCREENS:
                screens[s][m["name"]] = i
    end = b.n - 1
    return (
        weather[0] if end - weather[1] < FIELD_TURNS else None,
        {s: {k for k, j in v.items() if end - j < FIELD_TURNS} for s, v in screens.items()},
    )


def _spent(b: Battle, s: str) -> tuple[dict[str, int], dict[str, str]]:
    """Entries each Pokémon has spent in its current sleep or freeze, and the kind of spell."""
    side = b.sides[s]
    spent: dict[str, int] = {}
    kind: dict[str, str] = {}
    prev: dict[str, str] = {}
    for i in range(b.n):
        n, st = side.names[i], side.status[i]
        if st in ("slp", "frz"):
            if prev.get(n) == st:
                spent[n] += 1
            else:
                spent[n] = 1
                m = side.moves[i]
                rest = st == "slp" and m is not None and m["name"] == "rest"
                kind[n] = "rest" if rest else st
        prev[n] = st
    return spent, kind


def _fighter(
    n: str,
    side_key: str,
    hp: float,
    d: PokemonGen2,
    level: int,
    item: str | None,
    status: str,
    boosts: Mapping[str, int],
    revealed: Mapping[str, Move],
    table: MoveTable,
    delay: int,
) -> Fighter:
    moves = {m["name"]: m for m in revealed.values() if is_attack(m)}
    for m in table.typical.get(n, []) + proxies(d.types):
        moves.setdefault(m["name"], m)
    return {
        "name": n,
        "side": side_key,
        "hp": hp,
        "d": d,
        "level": level,
        "item": item,
        "status": status,
        "boosts": boosts,
        "moves": list(moves.values()),
        "delay": delay,
        "spe": ed.effective_speed(d, boosts, status, level),
    }


def _fighters2(
    b: Battle,
    side_key: str,
    track: Track,
    alive: Callable[[str], bool],
    table: MoveTable,
    active: str,
) -> list[Fighter]:
    side, i = b.sides[side_key], b.n - 1
    spent, kind = _spent(b, side_key)
    out: list[Fighter] = []
    for n, hp in track["hp"].items():
        d = b.species(n)
        if d is None or hp <= 0 or not alive(n):
            continue
        revealed, st = side.revealed.get(n, {}), track["status"][n]
        delay = 0
        if st == "slp" and "sleeptalk" not in revealed:
            delay = max(0, table.turns[kind.get(n, "slp")] - spent.get(n, 0))
        elif st == "frz":
            delay = max(0, table.turns["frz"] - spent.get(n, 0))
        boosts = side.boosts[i] if n == active else {}
        level, item = b.level(side_key, n), b.item(side_key, n)
        out.append(_fighter(n, side_key, hp, d, level, item, st, boosts, revealed, table, delay))
    return out


def _hidden(b: Battle, table: MoveTable, team_model: TeamModel) -> list[Fighter]:
    """Most likely unrevealed Player 2 Pokémon, at full HP."""
    revealed = list(b.sides["p2"].revealed)
    n_hidden = 6 - len(revealed)
    if n_hidden <= 0:
        return []
    e = team_model.expected_unrevealed(revealed, n_hidden)
    likely = ((n, b.species(n)) for n, _ in sorted(e.items(), key=lambda kv: -kv[1]))
    found = [(n, d) for n, d in likely if d is not None][:n_hidden]
    return [_fighter(n, "p2", 1.0, d, 100, None, "nostatus", {}, {}, table, 0) for n, d in found]


def _damage2(u: Fighter, v: Fighter, field: Field | None) -> float:
    best = 0.0
    for m in u["moves"]:
        x = ed.expected_damage(
            CHART,
            m,
            u["d"],
            v["d"],
            att_level=u["level"],
            def_level=v["level"],
            att_boosts=u["boosts"],
            def_boosts=v["boosts"],
            att_item=u["item"],
            att_status=u["status"],
            att_name=u["name"],
        )
        if field is not None:
            weather, screens = field
            x *= WEATHER_BOOST.get(weather, {}).get(m["type"], 1.0)
            if SCREEN_OF.get(m["category"]) in screens[v["side"]]:
                x *= 0.5
        best = max(best, x)
    return best


def _need(u: Fighter, v: Fighter, cache: Cache, field: Field | None = None) -> int:
    """Turns u needs to knock v out: hits plus the turns it still loses asleep or frozen."""
    key = (u["name"], v["name"])
    if key not in cache:
        cache[key] = _damage2(u, v, field)
    dmg = cache[key]
    return min(MAX_HITS, math.ceil(v["hp"] / dmg) + u["delay"]) if dmg > 0 else MAX_HITS


def _duel2(a: Fighter, b: Fighter, cache: Cache) -> tuple[bool, float]:
    ta, tb = _need(a, b, cache), _need(b, a, cache)
    a_fast = a["spe"] > b["spe"]
    if ta < tb or (ta == tb and a_fast):
        hits = max(0, (ta - 1 if a_fast else ta) - b["delay"])
        return True, max(MIN_HP, a["hp"] - hits * cache[b["name"], a["name"]])
    hits = max(0, (tb if a_fast else tb - 1) - a["delay"])
    return False, max(MIN_HP, b["hp"] - hits * cache[a["name"], b["name"]])


def _answer2(team: list[Fighter], foe: Fighter, cache: Cache) -> int:
    return max(
        range(len(team)),
        key=lambda k: (
            math.log(_need(foe, team[k], cache) / _need(team[k], foe, cache)),
            team[k]["spe"],
        ),
    )


def simulate2(
    A: list[Fighter], B: list[Fighter], act_a: str, act_b: str
) -> tuple[list[Fighter], list[Fighter]]:
    """The fighters left on each side after the chain of duels started by the actives."""
    A, B, cache = list(A), list(B), {}
    found_a = next((k for k, u in enumerate(A) if u["name"] == act_a), None)
    found_b = next((k for k, v in enumerate(B) if v["name"] == act_b), None)
    if not A or not B:
        return A, B
    ia = found_a if found_a is not None else _answer2(A, B[found_b or 0], cache)
    ib = found_b if found_b is not None else _answer2(B, A[ia], cache)
    for _ in range(MAX_DUELS):
        if not A or not B:
            break
        a_wins, hp = _duel2(A[ia], B[ib], cache)
        if a_wins:
            A[ia] = {**A[ia], "hp": hp, "delay": 0}
            B.pop(ib)
            if B:
                ib = _answer2(B, A[ia], cache)
        else:
            B[ib] = {**B[ib], "hp": hp, "delay": 0}
            A.pop(ia)
            if A:
                ia = _answer2(A, B[ib], cache)
    return A, B


def simulation_features(
    b: Battle, t1: Track, t2: Track, table: MoveTable, team_model: TeamModel
) -> dict[str, Any]:
    """The ``z_`` columns of a battle."""
    i = b.n - 1
    n1, n2 = b.sides["p1"].names[i], b.sides["p2"].names[i]
    A = _fighters2(b, "p1", t1, lambda n: t1["pf"][n] < 0.5, table, n1)
    B = _fighters2(b, "p2", t2, lambda n: t2["pf"][n] < 0.5, table, n2)
    a = next((u for u in A if u["name"] == n1), None)
    v = next((u for u in B if u["name"] == n2), None)
    duel = 0.0
    if a is not None and v is not None:
        field, cache = _field(b), {}
        ta, tv = _need(a, v, cache, field), _need(v, a, cache, field)
        if ta == tv:
            ta, tv = v["spe"], a["spe"]
        duel = float(ta < tv) - float(ta > tv)
    f: dict[str, Any] = {"z_act_duel": duel}
    for tag, foes in (("seen", B), ("full", B + _hidden(b, table, team_model))):
        left_a, left_b = simulate2(A, foes, n1, n2)
        f[f"z_sim_{tag}_left"] = (len(left_a) - len(left_b)) / 6
        f[f"z_sim_{tag}_hp"] = (sum(u["hp"] for u in left_a) - sum(u["hp"] for u in left_b)) / 6
    return f
