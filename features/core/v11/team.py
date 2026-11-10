"""v11's team identifiers: teams are reused across battles, so they identify (groups of) players.
A copy of the original's ``features/team.py``."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from itertools import combinations
from typing import Any

from features.core.v11.battle import Battle

NAN = float("nan")


def team_id(names: Iterable[str]) -> float:
    """Stable numeric id of a set of species (exact in float64)."""
    return float(int(hashlib.md5("|".join(sorted(names)).encode()).hexdigest()[:12], 16))


def team_keys(b: Battle) -> dict[str, Any]:
    """The ``k_`` columns: team, sub-team and species ids of both players.

    Ids of the P1 team, of the P2 team when all six were revealed (or of the five revealed ones),
    of the five-species subsets of both full teams, used to match a partially revealed P2 team to
    a known one, and of each species of both teams (P2: the revealed ones), used to find all the
    known teams that contain them. The ids are only used through ``TeamEffect``.
    """
    p1, rev2 = [p["name"] for p in b.team], list(b.sides["p2"].revealed)
    f: dict[str, Any] = {
        "k_p1team": team_id(p1),
        "k_p2team": team_id(rev2) if len(rev2) == 6 else NAN,
        "k_p2part": team_id(rev2) if len(rev2) == 5 else NAN,
    }
    subs1 = [team_id(s) for s in combinations(sorted(p1), 5)]
    subs2 = [team_id(s) for s in combinations(sorted(rev2), 5)] if len(rev2) == 6 else [NAN] * 6
    sp1 = sorted(team_id([n]) for n in p1)
    sp2 = sorted(team_id([n]) for n in rev2)
    for k in range(6):
        f[f"k_p1sub{k}"], f[f"k_p2sub{k}"] = subs1[k], subs2[k]
        f[f"k_p1sp{k}"], f[f"k_p2sp{k}"] = sp1[k], sp2[k] if k < len(sp2) else NAN
    return f
