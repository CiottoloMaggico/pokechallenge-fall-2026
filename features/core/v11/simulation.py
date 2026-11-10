"""v11's duel between the two actives and a quick simulation of the rest of the battle as a
sequence of one-on-one duels, with the HP carried over and the best answer sent in after each
knockout (first version: revealed plus typical moves of each species, or revealed moves only).

A copy of the first half of the original's ``features/simulation.py``; the second version
(``simulation_features``) is in ``simulation2.py``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from domain import expected_damage as ed
from domain.timeline import Move
from features.core.v11 import moves as mx
from features.core.v11.battle import Battle
from features.core.v11.matchup import DISABLE, Fighter, proxies
from features.core.v11.moves import CHART
from features.core.v11.state import Track

MAX_HITS = 30
MAX_DUELS = 15
MIN_HP = 0.01
TYPICAL_MOVES, MIN_COUNT, MIN_SHARE = 4, 3, 0.04

Cache = dict[tuple[str, str], float]


def is_attack(m: Move) -> bool:
    """A damaging move that is not a self-KO (fixed-damage moves included)."""
    return (
        m["category"] != "status"
        and m["name"] not in mx.SELF_KO
        and (m["base_power"] > 1 or m["name"] in mx.FIXED_DAMAGE)
    )


class MoveTable:
    """Typical attacking moves of each species and typical spell lengths.

    The typical moves are the most often revealed ones in the training battles; the spell lengths
    are the typical number of entries spent active asleep after Rest, in other sleep and frozen.
    """

    def __init__(self, battles: Sequence[Battle]) -> None:
        counts: dict[str, Counter[str]] = {}
        known: dict[str, Move] = {}
        for b in battles:
            for side in b.sides.values():
                for name, moves in side.revealed.items():
                    for m in moves.values():
                        if is_attack(m):
                            counts.setdefault(name, Counter())[m["name"]] += 1
                            known[m["name"]] = m
        self.typical = {
            name: [
                known[m]
                for m, n in c.most_common(TYPICAL_MOVES)
                if n >= MIN_COUNT and n / sum(c.values()) >= MIN_SHARE
            ]
            for name, c in counts.items()
        }
        self.turns = _spell_lengths(battles)


def _spell_lengths(battles: Sequence[Battle]) -> dict[str, int]:
    lengths: dict[str, list[int]] = {"rest": [], "slp": [], "frz": []}
    for b in battles:
        for side in b.sides.values():
            run, kind = 0, ""
            for i in range(b.n):
                st, same = side.status[i], i > 0 and side.names[i - 1] == side.names[i]
                if st in ("slp", "frz") and same and side.status[i - 1] == st:
                    run += 1
                elif st in ("slp", "frz"):
                    m = side.moves[i]
                    rest = st == "slp" and m is not None and m["name"] == "rest"
                    run, kind = 1, "rest" if rest else st
                else:
                    if run and same and side.status[i - 1] in ("slp", "frz"):
                        lengths[kind].append(run)
                    run, kind = 0, ""
    return {k: round(sum(v) / len(v)) if v else 2 for k, v in lengths.items()}


def _fighters(
    b: Battle,
    side_key: str,
    track: Track,
    alive: Callable[[str], bool],
    table: MoveTable | None,
    active: str,
) -> list[Fighter]:
    side, i = b.sides[side_key], b.n - 1
    out: list[Fighter] = []
    for n, hp in track["hp"].items():
        d = b.species(n)
        if d is None or hp <= 0 or not alive(n):
            continue
        moves = {m["name"]: m for m in side.revealed.get(n, {}).values() if is_attack(m)}
        for m in table.typical.get(n, []) if table else []:
            moves.setdefault(m["name"], m)
        st = track["status"][n]
        boosts = side.boosts[i] if n == active else {}
        level = b.level(side_key, n)
        mult = 1.0
        if st in DISABLE and not (st == "slp" and "sleeptalk" in side.revealed.get(n, {})):
            mult = DISABLE[st]
        out.append(
            {
                "name": n,
                "hp": hp,
                "d": d,
                "level": level,
                "item": b.item(side_key, n),
                "status": st,
                "boosts": boosts,
                "mult": mult,
                "moves": list(moves.values()) or proxies(d.types),
                "spe": ed.effective_speed(d, boosts, st, level),
            }
        )
    return out


def _damage(u: Fighter, v: Fighter, cache: Cache) -> float:
    key = (u["name"], v["name"])
    if key not in cache:
        cache[key] = u["mult"] * ed.best_damage(
            CHART,
            u["moves"],
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
    return cache[key]


def _hits(u: Fighter, v: Fighter, cache: Cache) -> int:
    dmg = _damage(u, v, cache)
    return min(MAX_HITS, math.ceil(v["hp"] / dmg)) if dmg > 0 else MAX_HITS


def _duel(a: Fighter, b: Fighter, cache: Cache) -> tuple[bool, float]:
    """(a wins, HP left to the winner)."""
    ha, hb = _hits(a, b, cache), _hits(b, a, cache)
    a_fast = a["spe"] > b["spe"]
    if ha < hb or (ha == hb and a_fast):
        return True, max(MIN_HP, a["hp"] - (ha - 1 if a_fast else ha) * _damage(b, a, cache))
    return False, max(MIN_HP, b["hp"] - (hb if a_fast else hb - 1) * _damage(a, b, cache))


def _best_answer(team: list[Fighter], foe: Fighter, cache: Cache) -> int:
    return max(
        range(len(team)),
        key=lambda k: math.log(_hits(foe, team[k], cache) / _hits(team[k], foe, cache)),
    )


def simulate(
    A: list[Fighter], B: list[Fighter], act_a: str, act_b: str
) -> tuple[list[Fighter], list[Fighter]]:
    """The fighters left on each side after the chain of duels started by the actives."""
    A, B, cache = list(A), list(B), {}
    found_a = next((k for k, u in enumerate(A) if u["name"] == act_a), None)
    found_b = next((k for k, v in enumerate(B) if v["name"] == act_b), None)
    if not A or not B:
        return A, B
    ia = found_a if found_a is not None else _best_answer(A, B[found_b or 0], cache)
    ib = found_b if found_b is not None else _best_answer(B, A[ia], cache)
    for _ in range(MAX_DUELS):
        if not A or not B:
            break
        a_wins, hp = _duel(A[ia], B[ib], cache)
        if a_wins:
            A[ia] = {**A[ia], "hp": hp}
            B.pop(ib)
            if B:
                ib = _best_answer(B, A[ia], cache)
        else:
            B[ib] = {**B[ib], "hp": hp}
            A.pop(ia)
            if A:
                ia = _best_answer(A, B[ib], cache)
    return A, B


def _outcomes(
    b: Battle, t1: Track, t2: Track, table: MoveTable | None, suffix: str
) -> dict[str, Any]:
    i = b.n - 1
    n1, n2 = b.sides["p1"].names[i], b.sides["p2"].names[i]
    A = _fighters(b, "p1", t1, lambda n: t1["pf"][n] < 0.5, table, n1)
    B = _fighters(b, "p2", t2, lambda n: t2["pf"][n] < 0.5, table, n2)
    f: dict[str, Any] = {}
    a = next((u for u in A if u["name"] == n1), None)
    v = next((u for u in B if u["name"] == n2), None)
    if a is None or v is None:
        f["d_act_duel" + suffix] = 0.0
    else:
        cache: Cache = {}
        ha, hb = _hits(a, v, cache), _hits(v, a, cache)
        if ha == hb:
            ha, hb = v["spe"], a["spe"]
        f["d_act_duel" + suffix] = float(ha < hb) - float(ha > hb)
    left_a, left_b = simulate(A, B, n1, n2)
    f["d_sim_left" + suffix] = (len(left_a) - len(left_b)) / 6
    f["d_sim_hp" + suffix] = (sum(u["hp"] for u in left_a) - sum(u["hp"] for u in left_b)) / 6
    return f


def duel_features(b: Battle, t1: Track, t2: Track, table: MoveTable) -> dict[str, Any]:
    """The ``d_`` columns: with the typical moves, then with the revealed moves only (``_seen``)."""
    return {**_outcomes(b, t1, t2, table, ""), **_outcomes(b, t1, t2, None, "_seen")}
