"""v11's switching behaviour: how likely each player is to switch out in the current matchup, how
much each player switched beyond those rates, the share of attacks thrown into resistances, and
Substitute and trapping differences between the two actives.

The rates of voluntary switches per pair of actives are learned from the training battles without
labels. A copy of the original's ``features/switching.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from domain import expected_damage as ed
from features.core.v11 import moves as mx
from features.core.v11.battle import OPP, Battle
from features.core.v11.moves import CHART

SMOOTHING = 5.0
TYPELESS = {"hiddenpower", "struggle"}
TRAPS = {"trapped"} | mx.PARTIAL_TRAP


def _turns(b: Battle, s: str) -> list[tuple[int, int, float]]:
    """(previous index, current index, voluntary switch weight) for the turns where side `s` could
    choose. Entries right after a Player 1 faint are replacement choices, not played turns."""
    own, opp, p1 = b.sides[s], b.sides[OPP[s]], b.sides["p1"]
    out: list[tuple[int, int, float]] = []
    for i in range(1, b.n):
        if p1.status[i - 1] == "fnt" or own.status[i - 1] == "fnt" or own.hp[i - 1] <= 0:
            continue
        w = 0.0
        if own.names[i] != own.names[i - 1]:
            m = opp.moves[i]
            phazed = m is not None and m["name"] in mx.PHAZE and opp.names[i] == opp.names[i - 1]
            w = 0.0 if phazed else 1.0 - (b.p2_faint.get(i, 0.0) if s == "p2" else 0.0)
        out.append((i - 1, i, w))
    return out


class SwitchRates:
    """Voluntary switch rate for each (own active, opposing active) pair, smoothed towards the
    rate of the own species and then the global rate."""

    def __init__(self, battles: Sequence[Battle]) -> None:
        pairs: dict[tuple[str, str], list[Any]] = {}
        for b in battles:
            for s in ("p1", "p2"):
                own, opp = b.sides[s], b.sides[OPP[s]]
                for j, _, w in _turns(b, s):
                    c = pairs.setdefault((own.names[j], opp.names[j]), [0.0, 0])
                    c[0] += w
                    c[1] += 1
        species: dict[str, list[Any]] = {}
        for (mine, _), (sw, n) in pairs.items():
            c = species.setdefault(mine, [0.0, 0])
            c[0] += sw
            c[1] += n
        tot = [sum(c[k] for c in species.values()) for k in (0, 1)]
        self.glob = tot[0] / tot[1]
        self.species = {
            k: (sw + SMOOTHING * self.glob) / (n + SMOOTHING) for k, (sw, n) in species.items()
        }
        self.pairs = {
            k: (sw + SMOOTHING * self.species[k[0]]) / (n + SMOOTHING)
            for k, (sw, n) in pairs.items()
        }

    def rate(self, mine: str, theirs: str) -> float:
        """Smoothed switch rate of ``mine`` facing ``theirs``."""
        return self.pairs.get((mine, theirs), self.species.get(mine, self.glob))


def _resisted_share(b: Battle, s: str) -> float:
    own, opp = b.sides[s], b.sides[OPP[s]]
    attacks = resisted = 0
    for i in range(1, b.n):
        m = own.moves[i]
        if (
            m is None
            or m["category"] == "status"
            or own.names[i] != own.names[i - 1]
            or own.status[i] in ("slp", "frz")
        ):
            continue
        d = b.species(opp.names[i])
        if d is None:
            continue
        attacks += 1
        resisted += m["name"] not in TYPELESS and ed.effectiveness(CHART, m["type"], d.types) < 1
    return resisted / attacks if attacks else 0.0


def behaviour_features(b: Battle, rates: SwitchRates) -> dict[str, Any]:
    """The ``v_`` columns of a battle."""
    last = b.n - 1
    prob: dict[str, float] = {}
    excess: dict[str, float] = {}
    for s in ("p1", "p2"):
        own, opp = b.sides[s], b.sides[OPP[s]]
        prob[s] = 1.0 if own.status[last] == "fnt" else rates.rate(own.names[last], opp.names[last])
        excess[s] = sum(w - rates.rate(own.names[j], opp.names[j]) for j, _, w in _turns(b, s))
    e1, e2 = b.sides["p1"].effects[last], b.sides["p2"].effects[last]
    return {
        "v_p_switch_diff": prob["p2"] - prob["p1"],
        "v_switch_excess": (excess["p2"] - excess["p1"]) / 30,
        "v_mv_res": _resisted_share(b, "p2") - _resisted_share(b, "p1"),
        "v_sub_diff": float("substitute" in e1) - float("substitute" in e2),
        "v_trap_diff": float(bool(TRAPS & e2)) - float(bool(TRAPS & e1)),
    }
