"""v11's field and team condition at the last decision point: Spikes on each side (re-laid and
spun ones included), stat stages of the actives (the Speed drop of Curse not counted), healthy
and statused shares, Snorlax's HP (expected from team co-occurrences while unrevealed) and the
physical and special bulk left (HP x Defense). A copy of the original's
``features/condition.py``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from domain import expected_damage as ed
from features.core.v11.replay import OPP, STATUSES, Battle
from features.core.v11.state import Track
from features.core.v11.team_model import TeamModel

SNORLAX = "snorlax"


def _spikes(b: Battle) -> dict[str, bool]:
    on = {"p1": False, "p2": False}
    for i in range(b.n):
        for s in ("p1", "p2"):
            m = b.timeline.used_move(s, i)
            if m is None:
                continue
            if m["name"] == "spikes":
                on[OPP[s]] = True
            elif m["name"] == "rapidspin":
                foe = b.species(b.sides[OPP[s]].names[i])
                if foe is None or "ghost" not in foe.types:
                    on[s] = False
    return on


def _boost_total(bo: Mapping[str, int]) -> int:
    total = sum(bo.values())
    return total - bo["spe"] if bo["atk"] > 0 and bo["def"] > 0 and bo["spe"] < 0 else total


def _bulk(
    b: Battle, side_key: str, track: Track, weight: Callable[[str], float], stat: str
) -> float:
    total, full = 0.0, []
    for n, hp in track["hp"].items():
        d = b.species(n)
        if d is None:
            continue
        level = b.level(side_key, n)
        size = ed.stat_value(d.base_hp, level, hp=True) * ed.stat_value(d.base(stat), level)
        full.append(size)
        total += weight(n) * hp * size
    if side_key == "p2" and full:
        total += (6 - len(track["hp"])) * sum(full) / len(full)
    return total


def condition_features(b: Battle, t1: Track, t2: Track, team_model: TeamModel) -> dict[str, Any]:
    """The ``c_`` columns of a battle."""
    i = b.n - 1
    on = _spikes(b)

    def w1(n: str) -> float:
        return 1 - t1["pf"][n]

    def w2(n: str) -> float:
        return 1 - t2["pf"][n]

    f: dict[str, Any] = {
        "c_spikes": float(on["p2"]) - float(on["p1"]),
        "c_boosts": (_boost_total(b.sides["p1"].boosts[i]) - _boost_total(b.sides["p2"].boosts[i]))
        / 30,
    }
    unrevealed = 6 - len(t2["hp"])
    healthy1 = sum(w1(n) for n in t1["hp"] if t1["status"][n] == "nostatus")
    healthy2 = sum(w2(n) for n in t2["hp"] if t2["status"][n] == "nostatus") + unrevealed
    f["c_healthy"] = (healthy1 - healthy2) / 6
    f["c_statused"] = (
        sum(w1(n) for n in t1["hp"] if t1["status"][n] in STATUSES)
        - sum(w2(n) for n in t2["hp"] if t2["status"][n] in STATUSES)
    ) / 6
    snorlax1 = w1(SNORLAX) * t1["hp"][SNORLAX] if SNORLAX in t1["hp"] else 0.0
    if SNORLAX in t2["hp"]:
        snorlax2 = w2(SNORLAX) * t2["hp"][SNORLAX]
    else:
        snorlax2 = team_model.expected_unrevealed(list(t2["hp"]), unrevealed).get(SNORLAX, 0.0)
    f["c_snorlax"] = snorlax1 - snorlax2
    for stat in ("def", "spd"):
        b1, b2 = _bulk(b, "p1", t1, w1, stat), _bulk(b, "p2", t2, w2, stat)
        f[f"c_bulk_{stat}"] = (b1 - b2) / (b1 + b2) if b1 + b2 > 0 else 0.0
    return f
