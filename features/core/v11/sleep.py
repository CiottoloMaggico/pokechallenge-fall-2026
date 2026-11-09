"""v11's sleep features: longest current sleep, Rest and induced sleepers of each player. A copy
of ``sleep_features`` of the original's ``features/extra.py`` (its usage bags are not ported)."""

from __future__ import annotations

from typing import Any

from features.core.v11.replay import Battle
from features.core.v11.state import Track


def sleep_features(b: Battle, t1: Track, t2: Track) -> dict[str, Any]:
    """The ``x_`` columns of a battle."""
    f: dict[str, Any] = {}
    for key, track in (("p1", t1), ("p2", t2)):
        side = b.sides[key]
        prev: dict[str, str] = {}
        spell: dict[str, list[Any] | None] = {}
        for i in range(b.n):
            n, st = side.names[i], side.status[i]
            if st == "slp":
                current = spell.get(n)
                if prev.get(n) != "slp" or current is None:
                    m = side.moves[i]
                    current = spell[n] = [0, m is not None and m["name"] == "rest"]
                current[0] += 1
            else:
                spell[n] = None
            prev[n] = st
        spells = {n: sp for n, sp in spell.items() if sp is not None}
        sleepers = [n for n in spells if track["pf"].get(n, 0) < 0.5]
        f[f"x_{key}_sleep_turns_max"] = max((spells[n][0] for n in sleepers), default=0)
        f[f"x_{key}_rest_sleepers"] = sum(spells[n][1] for n in sleepers)
        f[f"x_{key}_induced_sleepers"] = sum(not spells[n][1] for n in sleepers)
    for k in ("sleep_turns_max", "rest_sleepers", "induced_sleepers"):
        f[f"x_d_{k}"] = f[f"x_p1_{k}"] - f[f"x_p2_{k}"]
    return f
